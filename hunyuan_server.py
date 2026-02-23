#!/usr/bin/env python3
"""HunyuanImage-3.0 generation server.

Loads the model once into GPU memory and exposes an HTTP API for generation.
Single-threaded generation (one GPU set = one job at a time), serialized with
a threading lock.

Endpoints:
    GET  /health      — Returns model info, port, PID.
    POST /generate    — Accepts JSON, returns JSON with output path + metadata.
    POST /understand  — Accepts JSON (prompt + image), returns JSON with text response.

Usage:
    python hunyuan_server.py --model-id /raid/weights/HunyuanImage-3-Instruct --port 8079
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
import traceback
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import torch

# Project imports (PYTHONPATH must include the project root)
from hunyuan_image_3 import HunyuanImage3ForCausalMM
from hunyuan_image_3.hunyuan_image_3_pipeline import FlowMatchDiscreteScheduler
from hunyuan_image_3.system_prompt import get_system_prompt as resolve_system_prompt

# ---------------------------------------------------------------------------
# Globals set during startup
# ---------------------------------------------------------------------------
model = None
model_name = None
server_port = None
gen_lock = threading.Lock()


def parse_args():
    p = argparse.ArgumentParser(description="HunyuanImage-3.0 HTTP generation server")
    p.add_argument("--model-id", type=str, required=True, help="Path to model weights")
    p.add_argument("--port", type=int, default=8079, help="Port to listen on")
    p.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"])
    p.add_argument("--moe-impl", type=str, default="flashinfer", choices=["eager", "flashinfer"])
    return p.parse_args()


def load_model(args):
    """Load model + tokenizer, force pipeline creation, set reproducibility."""
    global model, model_name

    model_name = os.path.basename(args.model_id)
    print(f"Loading model: {model_name}", flush=True)
    t0 = time.time()

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
        moe_drop_tokens=True,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    # Force pipeline + scheduler creation
    _ = model.pipeline

    # Set reproducibility defaults (same as run_image_gen.py)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    elapsed = time.time() - t0
    print(f"Config: cfg_distilled={model.config.cfg_distilled}, "
          f"use_meanflow={model.config.use_meanflow}, "
          f"model_version={getattr(model.config, 'model_version', 'unknown')}", flush=True)
    print(f"Model loaded in {elapsed:.1f}s", flush=True)


def ensure_scheduler(solver="euler", flow_shift=3.0):
    """Swap scheduler if solver or flow_shift differ from current config."""
    sched = model.pipeline.scheduler
    current_solver = sched.config.get("solver", "euler")
    current_shift = sched.config.get("shift", 3.0)
    if current_solver != solver or current_shift != flow_shift:
        new = FlowMatchDiscreteScheduler(shift=flow_shift, reverse=True, solver=solver,
                                         meanflow=getattr(model.config, 'use_meanflow', False))
        model.scheduler = new
        model.pipeline.set_scheduler(new)
        print(f"Scheduler swapped: solver={solver}, shift={flow_shift}", flush=True)


# Mode → bot_task mapping
MODE_MAP = {
    "think": "think_recaption",
    "rewrite": "recaption",
    "direct": "image",
}

# Valid system prompt presets
VALID_SYSTEM_PROMPTS = {"en_vanilla", "en_recaption", "en_think_recaption", "en_unified", "dynamic", "likeness", "None"}

# Valid solvers
VALID_SOLVERS = {"euler", "heun-2", "midpoint-2", "kutta-4"}

# Default system prompt for TI2T (visual intelligence) mode
TI2T_SYSTEM_PROMPT = (
    "You are an elite visual intelligence system capable of extraordinarily detailed "
    "image analysis. When the user provides an image along with a question or instruction, "
    "carefully examine the image and respond with a thorough, perceptive text answer in "
    "English. Do not generate images — respond with text only."
)


def do_generate(params):
    """Run a single generation. Called with gen_lock held."""
    prompt = params["prompt"]
    seed = params.get("seed")
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)

    size = params.get("size", "1024x1024")
    mode = params.get("mode", "think")
    solver = params.get("solver", "euler")
    steps = params.get("steps", None)  # None = use model default
    guidance_scale = params.get("guidance_scale", None)
    flow_shift = params.get("flow_shift", 3.0)
    system_prompt_preset = params.get("system_prompt_preset", None)
    system_prompt_text = params.get("system_prompt_text", None)
    image_path = params.get("image", None)
    temperature = params.get("temperature", None)
    top_k = params.get("top_k", None)
    top_p = params.get("top_p", None)
    verbose = params.get("verbose", 2)
    output = params.get("output", None)

    # Validate
    if mode not in MODE_MAP:
        raise ValueError(f"Invalid mode: {mode!r} (use think, rewrite, or direct)")
    if solver not in VALID_SOLVERS:
        raise ValueError(f"Invalid solver: {solver!r} (use {', '.join(sorted(VALID_SOLVERS))})")
    if system_prompt_preset is not None and system_prompt_preset not in VALID_SYSTEM_PROMPTS:
        raise ValueError(f"Invalid system_prompt_preset: {system_prompt_preset!r}")

    bot_task = MODE_MAP[mode]

    # Determine system prompt args
    use_system_prompt = None
    system_prompt = None
    if system_prompt_preset == "none" or system_prompt_preset == "None":
        use_system_prompt = "None"
    elif system_prompt_preset is not None:
        use_system_prompt = system_prompt_preset
    if system_prompt_text is not None:
        use_system_prompt = "custom"
        system_prompt = system_prompt_text

    # Swap scheduler if needed
    ensure_scheduler(solver=solver, flow_shift=flow_shift)

    # Set reproducibility seed, saving state to restore later
    # so we don't poison the global random state for subsequent seed=-1 calls
    import random as _random
    import numpy as np
    _rand_state = _random.getstate()
    _np_state = np.random.get_state()
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build kwargs
    gen_kwargs = dict(verbose=verbose, max_new_tokens=2048)
    if steps is not None:
        gen_kwargs["diff_infer_steps"] = steps
    if guidance_scale is not None:
        gen_kwargs["diff_guidance_scale"] = guidance_scale
    if temperature is not None:
        gen_kwargs["temperature"] = temperature
    if top_k is not None:
        gen_kwargs["top_k"] = top_k
    if top_p is not None:
        gen_kwargs["top_p"] = top_p

    # Handle input image
    image_input = None
    if image_path:
        image_paths = [p.strip() for p in image_path.split(",") if p.strip()]
        if len(image_paths) == 1:
            image_input = image_paths[0]
        elif len(image_paths) > 1:
            image_input = image_paths

    # Output path
    if output is None:
        output_dir = Path.home() / "images" / "hunyuan"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = str(output_dir / f"{timestamp}.png")
    else:
        Path(output).parent.mkdir(parents=True, exist_ok=True)

    # Generate
    t0 = time.time()
    cot_text, samples = model.generate_image(
        prompt=prompt,
        seed=seed,
        image_size=size,
        use_system_prompt=use_system_prompt,
        system_prompt=system_prompt,
        bot_task=bot_task,
        image=image_input,
        **gen_kwargs,
    )
    elapsed = time.time() - t0

    # Save
    samples[0].save(output)

    # Restore global random state
    _random.setstate(_rand_state)
    np.random.set_state(_np_state)

    # Memory cleanup
    gc.collect()
    torch.cuda.empty_cache()

    # Build actual params used (for response)
    actual_params = {
        "solver": solver,
        "steps": steps if steps is not None else getattr(model.generation_config, "diff_infer_steps", 50),
        "guidance_scale": guidance_scale if guidance_scale is not None else getattr(model.generation_config, "diff_guidance_scale", 2.5),
        "flow_shift": flow_shift,
        "mode": mode,
        "size": size,
        "system_prompt_preset": system_prompt_preset or "dynamic",
    }

    cot_str = None
    if cot_text and isinstance(cot_text, list) and cot_text[0]:
        cot_str = cot_text[0]

    return {
        "status": "ok",
        "file": output,
        "seed": seed,
        "cot_text": cot_str,
        "elapsed": round(elapsed, 1),
        "params": actual_params,
    }


def do_understand(params):
    """Run a TI2T (visual intelligence) query. Called with gen_lock held."""
    prompt = params["prompt"]
    image_path = params.get("image")
    if not image_path:
        raise ValueError("'image' is required for /understand")

    seed = params.get("seed")
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)

    temperature = params.get("temperature", None)
    top_k = params.get("top_k", None)
    top_p = params.get("top_p", None)
    max_new_tokens = params.get("max_new_tokens", 2048)
    system_prompt_preset = params.get("system_prompt_preset", None)
    system_prompt_text = params.get("system_prompt_text", None)
    verbose = params.get("verbose", 2)

    # Build message list: image first, then text prompt
    image_paths = [p.strip() for p in image_path.split(",") if p.strip()]
    message_list = []
    for p in image_paths:
        message_list.append({"role": "user", "content": [{"type": "image", "path": p}]})
    message_list.append({"role": "user", "content": prompt})

    # Resolve system prompt to actual text
    # If user explicitly chose a preset, use it. Otherwise default to TI2T prompt.
    if system_prompt_text is not None:
        system_prompt = system_prompt_text.strip()
    elif system_prompt_preset is not None:
        system_prompt = resolve_system_prompt(system_prompt_preset, "auto", None)
        if system_prompt is not None:
            system_prompt = system_prompt.strip()
    else:
        system_prompt = TI2T_SYSTEM_PROMPT

    # Set reproducibility seed, saving state to restore later
    import random as _random
    import numpy as np
    _rand_state = _random.getstate()
    _np_state = np.random.get_state()
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Prepare model inputs for text generation
    t0 = time.time()
    model_inputs = model.prepare_model_inputs(
        message_list=message_list,
        seed=seed,
        image_size="auto",
        mode="gen_text",
        bot_task="auto",
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
    )
    input_length = model_inputs["input_ids"].shape[1]
    model_inputs["verbose"] = verbose
    model_inputs["decode_text"] = True

    # Pass sampling params directly to generate() so they override generation_config
    if temperature is not None:
        model_inputs["temperature"] = temperature
    if top_k is not None:
        model_inputs["top_k"] = top_k
    if top_p is not None:
        model_inputs["top_p"] = top_p

    # Generate text (blocking, no streamer needed)
    output = model.generate(**model_inputs)
    elapsed = time.time() - t0

    # output is a list of strings when decode_text=True
    text = output[0] if isinstance(output, list) else output

    # Clean up special tokens from output
    for tok in ("<|endoftext|>", "</s>"):
        text = text.replace(tok, "")
    text = text.strip()

    # Restore global random state
    _random.setstate(_rand_state)
    np.random.set_state(_np_state)

    # Memory cleanup
    del model_inputs
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "status": "ok",
        "text": text,
        "seed": seed,
        "elapsed": round(elapsed, 1),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress default access logs
        pass

    def _send_json(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model": model_name,
                "port": server_port,
                "pid": os.getpid(),
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/generate", "/understand"):
            self._send_json(404, {"error": "not found"})
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "empty request body"})
            return
        raw = self.rfile.read(content_length)
        try:
            params = json.loads(raw)
        except json.JSONDecodeError as e:
            self._send_json(400, {"error": f"invalid JSON: {e}"})
            return

        if "prompt" not in params:
            self._send_json(400, {"error": "missing required field: prompt"})
            return

        # Serialize generation
        acquired = gen_lock.acquire(timeout=0)
        if not acquired:
            self._send_json(503, {"error": "generation already in progress, try again later"})
            return

        try:
            if self.path == "/understand":
                print(f"[understand] prompt={params['prompt'][:80]!r} ...", flush=True)
                result = do_understand(params)
                print(f"[understand] done in {result['elapsed']}s ({len(result['text'])} chars)", flush=True)
            else:
                print(f"[generate] prompt={params['prompt'][:80]!r} ...", flush=True)
                result = do_generate(params)
                print(f"[generate] done in {result['elapsed']}s -> {result['file']}", flush=True)
            self._send_json(200, result)
        except Exception as e:
            traceback.print_exc()
            self._send_json(500, {"error": str(e)})
        finally:
            gen_lock.release()


def main():
    global server_port
    args = parse_args()
    server_port = args.port

    load_model(args)

    server = ThreadingHTTPServer(("0.0.0.0", server_port), Handler)
    print(f"Server listening on port {server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
