# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import random
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import gradio as gr
from gradio import ChatMessage

from app.pipeline import HunyuanImage3AppPipeline
from app.style import load_css
from hunyuan_image_3.system_prompt import t2i_system_prompts

# Global vars
hyi3_pipeline: Optional[HunyuanImage3AppPipeline] = None
image_cache_dir: Optional[Path] = None


def default(val, default_val):
    return val if val is not None else default_val


def load_pipeline(args):
    """ Load the HunyuanImage-3 pipeline """
    global hyi3_pipeline
    hyi3_pipeline = HunyuanImage3AppPipeline(args)
    print("Model and tokenizer loaded.")

    global image_cache_dir
    image_cache_dir = args.image_cache_dir
    if image_cache_dir is not None:
        # Cache image by date
        image_cache_dir = Path(image_cache_dir)
        print("Image cache dir:", image_cache_dir)


def update_history(history, message):
    """ Update chatbot history """
    assert 'text' in message and 'files' in message

    # extra_img_input = preprocess_mask_img(img_input)
    extra_img_input = None
    for x in message["files"]:
        history.append(ChatMessage(role="user", content=gr.Image(x)))
    if message["text"] is not None:
        history.append(ChatMessage(role="user", content=message["text"]))
    if extra_img_input is not None:
        history.append(ChatMessage(role="user", content=gr.Image(extra_img_input)))
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def spinner():
    """ Return a spinner to denote image generation in progress """
    return """<div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>"""


def hunyuan_image_3_respond(history, system_prompt,
                            seed, top_k, top_p, temperature, infer_steps, diff_guidance_scale,
                            image_size, bot_task, context_mode,
                            ):
    """
    HunyuanImage-3 response generation function

    Args:
        history (List[Dict[str, str]]): Chat history
        system_prompt (str): System prompt
        seed (int): Random seed. -1 means random seed.
        top_k (float): Top-K for text generation
        top_p (float): Top-P for text generation:
        temperature (float): Temperature for text generation
        infer_steps (int): Diffusion inference steps
        diff_guidance_scale (float): Diffusion guidance scale
        image_size (str): Image size. "auto" or "HxW" or "H:W"
        bot_task (str): Bot task.
            "image": Only generate image. If image_size is "auto", the image ratio token will be predicted at first.
            "auto": Text generation. The model will decide whether to generate text or image.
            "think": Given user inputs, start thinking and then rewrite the prompt for image generation, finally
                        generate image.
            "recaption": Given user inputs, rewrite the prompt for image generation, finally generate image.
        context_mode (str): Context mode. "single_round", "unlimited"
    """
    extra_kwargs = {
        "seed": random.randint(0, 1_000_000) if seed < 0 else seed,
        "top_k": top_k,
        "top_p": top_p,
        "temperature": float(temperature),
        "diff_infer_steps": infer_steps,
        "diff_guidance_scale": diff_guidance_scale,
        "image_size": image_size,
        "bot_task": bot_task,
        "context_mode": context_mode,
        "drop_think": hyi3_pipeline.model.generation_config.drop_think,     # drop think when gen_image
    }
    eos = "<|endoftext|>"

    input_message_list = [message for message in history if message["content"] != ""]
    if system_prompt:
        input_message_list = [dict(
            role="system", content=system_prompt, type="text", content_type='str',
        )] + input_message_list

    current_text_response = ""
    history.append({"role": "assistant", "content": ""})
    last_yield_time = 0.0
    text_dirty = False

    for r in hyi3_pipeline.generate(input_message_list, **extra_kwargs):
        if r["type"] == "text" and r["value"] not in (eos, ""):
            current_text_response += r["value"]
            history[-1]["content"] = current_text_response
            text_dirty = True
            now = time.monotonic()
            if now - last_yield_time >= 0.05:
                yield history
                last_yield_time = time.monotonic()
                text_dirty = False

        elif r["type"] == "flag":
            if r["value"] == "image":
                # Flush any pending text before the image flag
                if text_dirty:
                    yield history
                    text_dirty = False
                if current_text_response:
                    current_text_response = ""
                history.append({"role": "assistant", "content": spinner()})
                yield history
                last_yield_time = time.monotonic()

        elif r["type"] == "image":
            # Flush any pending text before the image
            if text_dirty:
                yield history
                text_dirty = False
            if current_text_response:
                history.append({"role": "assistant", "content": ""})
                current_text_response = ""
            # Remove spinner
            if history[-1]["content"] == spinner():
                history.pop()
            # Save to /tmp for Gradio display (must be in allowed path).
            # Storing a PIL object in history causes Gradio to re-encode
            # it to PNG on every subsequent yield.
            fd, tmp_path = tempfile.mkstemp(suffix=".png", prefix="hunyuan_")
            os.close(fd)
            r["value"].save(tmp_path, format="PNG")
            history.append({"role": "assistant", "content": gr.Image(tmp_path)})
            if image_cache_dir is not None:
                date = datetime.now()
                cache_path = image_cache_dir / date.strftime("%Y%m%d") / f"img_{date.strftime('%H%M%S_%f')}.png"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_path, str(cache_path))
                print(f"Image saved to {cache_path}")
            yield history
            last_yield_time = time.monotonic()
            history.append({"role": "assistant", "content": ""})

    if not history[-1]["content"]:
        history.pop()
    yield history


def handle_undo(history, undo_data: gr.UndoData):
    """ Handle undo action """
    return history[:undo_data.index], gr.MultimodalTextbox(value=history[undo_data.index]['content'], interactive=True)


def handle_retry(history, retry_data: gr.RetryData, *args, **kwargs):
    """ Handle retry action """
    new_history = history[:retry_data.index + 1]
    yield from hunyuan_image_3_respond(new_history, *args, **kwargs)


def get_system_prompt(sys_type, bot_task):
    if sys_type == 'None':
        visible = False
        value = ""
    elif sys_type in ['en_vanilla', 'en_recaption', 'en_think_recaption']:
        visible = True
        value = t2i_system_prompts[sys_type][0]
    elif sys_type == "dynamic":
        visible = True
        if bot_task == "think":
            value = t2i_system_prompts["en_think_recaption"][0]
        elif bot_task == "recaption":
            value = t2i_system_prompts["en_recaption"][0]
        elif bot_task == "image":
            value = t2i_system_prompts["en_vanilla"][0].strip("\n")
        else:
            value = ""
    elif sys_type == 'custom':
        visible = True
        value = ""
    else:
        raise NotImplementedError(f"Unsupported system prompt type: {sys_type}")
    return gr.TextArea(value=value, lines=7, max_lines=7, placeholder="Please input system prompt", show_label=False,
                       visible=visible, elem_id="system-prompt")


def create_ui_interface(args):
    gen_config = hyi3_pipeline.model.generation_config
    block = gr.Blocks(fill_height=True, css=load_css())
    with block:
        with gr.Column():
            # ==== Left ====
            #  Sidebar
            with gr.Sidebar(open=args.open_sidebar, width='20%'):
                with gr.Accordion("Image Generation", open=True, visible=True):
                    with gr.Row(elem_id="Image Generation parameter", visible=True):
                        image_size = gr.Dropdown([
                            ("Auto", "auto"),
                            ("1:1", "1024x1024"),
                            ("4:3", "896x1152"),
                            ("3:4", "1152x896"),
                            ("16:9", "768x1280"),
                            ("9:16", "1280x768"),
                            ("21:9", "640x1408"),
                        ], label="Image size", value=args.image_size)
                        seed = gr.Number(
                            label="Seed", minimum=-1, maximum=1_000_000, value=args.seed, step=1, precision=0,
                            min_width=80,
                        )
                        infer_steps = gr.Slider(
                            label="Infer Steps", minimum=1, maximum=200,
                            value=default(args.diff_infer_steps, gen_config.diff_infer_steps), step=1,
                            min_width=200,
                        )
                        diff_guidance_scale = gr.Slider(
                            label="Guidance", minimum=1.0, maximum=16.0,
                            value=default(args.diff_guidance_scale, gen_config.diff_guidance_scale), step=0.5,
                            min_width=200,
                        )
                        use_system_prompt = gr.Dropdown([
                            ("None", 'None'),
                            ("Preset(Dynamic)", "dynamic"),
                            ("Preset(Default)", 'en_vanilla'),
                            ("Preset(Recaption)", 'en_recaption'),
                            ("Preset(Think+Recaption)", 'en_think_recaption'),
                            ("Custom", 'custom'),
                        ], label="System Prompt", value=default(args.use_system_prompt, gen_config.use_system_prompt))
                        bot_task = gr.Dropdown([
                            ("Image", "image"),
                            ("Auto", "auto"),
                            ("Think", "think"),
                            ("Recaption", "recaption"),
                        ], label="Bot Task", value=default(args.bot_task, gen_config.bot_task), min_width=150)
                        context_mode = gr.Dropdown([
                            ("Single Round", "single_round"),
                            ("All", "unlimited"),
                        ], label="Context Mode", value=args.context_mode, min_width=150)
                with gr.Accordion("Text Generation", open=False, visible=True):
                    with gr.Row(elem_id="Text Generation parameter"):
                        top_k = gr.Slider(
                            label="Top-K", minimum=1, maximum=16384,
                            value=default(args.top_k, gen_config.top_k), step=1, min_width=200,
                        )
                        top_p = gr.Slider(
                            label="Top-P", minimum=0.0, maximum=1.0,
                            value=default(args.top_p, gen_config.top_p), step=0.01, min_width=200,
                        )
                        temperature = gr.Slider(
                            label="Temperature", minimum=0.1, maximum=1.0,
                            value=default(args.temperature, gen_config.temperature), step=0.1, min_width=200,
                        )

            # ==== Right ====
            #  System prompt
            accordion = gr.Accordion("System Prompt", open=False)
            with accordion:
                system_prompt = get_system_prompt(
                    default(args.use_system_prompt, gen_config.use_system_prompt),
                    default(args.bot_task, gen_config.bot_task),
                )
            #  Chatbot
            chatbot = gr.Chatbot(
                min_height=500,
                elem_id="chatbot",
                bubble_full_width=False,
                type="messages",
                scale=1,
                avatar_images=('./assets/user.png', './assets/robot.png'),
                allow_tags=["think", "recaption"],
            )
            #  Input text box
            with gr.Row(scale=0):
                chat_input = gr.MultimodalTextbox(
                    interactive=True,
                    file_count="multiple",
                    file_types=["image"],
                    scale=15,
                    placeholder="Enter message or upload file...", show_label=False,
                    max_plain_text_length=65536,
                )

            #  Events
            chatbot.undo(handle_undo, chatbot, [chatbot, chat_input])
            chatbot.retry(
                handle_retry,
                [
                    chatbot, system_prompt,
                    seed, top_k, top_p, temperature, infer_steps, diff_guidance_scale,
                    image_size, bot_task, context_mode,
                ],
                chatbot,
            )

            chat_input.submit(
                update_history,
                [chatbot, chat_input],
                [chatbot, chat_input],
                queue=False,
            ).then(
                hunyuan_image_3_respond,
                [
                    chatbot, system_prompt,
                    seed, top_k, top_p, temperature, infer_steps, diff_guidance_scale,
                    image_size, bot_task, context_mode
                ],
                chatbot
            ).then(
                lambda: gr.MultimodalTextbox(interactive=True), None, [chat_input]
            )

            use_system_prompt.change(fn=get_system_prompt, inputs=[use_system_prompt, bot_task], outputs=system_prompt)
            bot_task.change(fn=get_system_prompt, inputs=[use_system_prompt, bot_task], outputs=system_prompt)

    return block


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    # server
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8080, help="Port to run the server on")
    parser.add_argument("--image-cache-dir", type=str, help="Directory where images are saved.")
    # ui
    parser.add_argument("--open-sidebar", action="store_true", help="Whether to open the sidebar by default")
    # model
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation")
    # inference
    parser.add_argument("--seed", type=int, default="-1", help="Random seed")
    parser.add_argument("--diff-infer-steps", type=int, help="Number of inference steps")
    parser.add_argument("--diff-guidance-scale", type=float, help="Guidance scale")
    parser.add_argument("--image-size", type=str, default="auto", help="Image size")
    parser.add_argument("--bot-task", type=str, choices=["image", "auto", "think", "recaption", "img_ratio"],
                        help="Bot task type for generating text.")
    parser.add_argument("--context-mode", type=str, default="single_round", choices=["single_round", "unlimited"],
                        help="Context mode")
    parser.add_argument("--top-k", type=int, help="Top-K")
    parser.add_argument("--top-p", type=float, help="Top-P")
    parser.add_argument("--temperature", type=float, help="Temperature")
    parser.add_argument("--use-system-prompt", type=str,
                        choices=["en_vanilla", "en_recaption", "en_think_recaption", "dynamic", "custom", "None"],
                        help="System prompt type")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_pipeline(args)

    chatbot_ui = create_ui_interface(args)
    chatbot_ui.launch(server_name=args.host, server_port=args.port, share=False)
