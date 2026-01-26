#!/usr/bin/env python3
"""
Parameterized client for calling vLLM-hunyuanimage3 text-to-image API
python openai_client.py --bot-task image --width 1024 --height 1024 --seed 42
"""

import argparse
import json
import requests
import base64
import random


def default(value, default_value):
    return value if value is not None else default_value


# ------------------ Default Parameters ------------------
DEFAULTS = {
    "prompt": "Generate an image: In a colosseum, a woman and a bear engage in combat, illuminated by torchlight. Rendered in 3D style.",
    "url": "http://0.0.0.0:8000/v1/chat/completions",
    "model": "vllm_hunyuan_image3",
    "max_tokens": 256,
    "temperature": 0,
}

# ------------------ Template Selection ------------------
TEMPLATES_PRETRAIN = {
    "image": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|startoftext|>{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "auto": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "<|startoftext|>{{ message['content'] }}<boi><image_shape_1024>"
        "{% endif %}"
        "{% endfor %}"
    ),
}


# ------------------ Main Logic ------------------
def build_payload(args):
    if args.sequence_template == "pretrain":
        templates = TEMPLATES_PRETRAIN
    else:
        raise NotImplementedError(f"Sequence template {args.sequence_template} not implemented.")

    chat_template = templates[args.bot_task]
    task_extra_kwargs = {
        "diff_infer_steps": args.diff_infer_steps,
        "use_system_prompt": args.use_system_prompt,
        "bot_task": args.bot_task,
    }
    if args.bot_task == "image":
        task_extra_kwargs["image_size"] = f"{default(args.height, 1024)}x{default(args.width, 1024)}"

    max_tokens = args.max_tokens
    if args.bot_task in ['image', 'auto']:
        max_tokens = 1

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": args.prompt}
        ],
        "max_completion_tokens": max_tokens,
        "temperature": args.temperature,
        "seed": default(args.seed, random.randint(1, 10_000_000)),
        "chat_template": chat_template,
        "task_type": "hunyuan_image3",
        "task_extra_kwargs": task_extra_kwargs,
    }
    return payload


def main():
    parser = argparse.ArgumentParser(description="Call vLLM-hunyuan_image3 text-to-image API")
    parser.add_argument("--sequence_template", choices=["pretrain", "instruct"], default="pretrain")
    parser.add_argument("--width", type=int, help="Image width")
    parser.add_argument("--height", type=int, help="Image height")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--prompt", default=DEFAULTS["prompt"])
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--use-system-prompt", type=str, default="None",
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Defaults to loading from model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, default="", 
                        help="Custom system prompt. Used when --use-system-prompt is 'custom'.")
    parser.add_argument("--bot-task", type=str, default="image", 
                        choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image. "
                             "Defaults to loading from model generation config.")
    parser.add_argument("--url", default=DEFAULTS["url"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--max_tokens", type=int, default=DEFAULTS["max_tokens"])
    parser.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])

    args = parser.parse_args()

    payload = build_payload(args)
    headers = {"Content-Type": "application/json"}

    resp = requests.post(args.url, data=json.dumps(payload), headers=headers, timeout=10000)
    print("Status:", resp.status_code)
    if resp.status_code != 200:
        print("Error:", resp.text)
        return
    
    data = resp.json()
    base64_image = data['image']

    # Remove possible data:image/png;base64, prefix
    if ',' in base64_image:
        base64_image = base64_image.split(',')[1]

    # Decode and save as PNG
    image_data = base64.b64decode(base64_image)
    with open("output.png", "wb") as f:
        f.write(image_data)
    print("Image saved as output.png")


if __name__ == "__main__":
    main()