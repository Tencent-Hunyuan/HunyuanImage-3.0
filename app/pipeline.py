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

import gc
import re
import time
from copy import deepcopy
from functools import wraps
from threading import Thread
from typing import List, Dict, Any, Optional

import gradio
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from hunyuan_image_3.tokenizer_wrapper import ImageInfo
from hunyuan_image_3.system_prompt import t2i_system_prompts


def with_inference_mode(func):
    """Decorator to wrap function with torch.inference_mode for thread-safe inference."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.inference_mode():
            return func(*args, **kwargs)
    return wrapper


def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class HunyuanImage3AppPipeline(object):
    def __init__(self, args):
        kwargs = dict(
            attn_implementation=args.attn_impl,
            torch_dtype="auto",
            device_map="auto",
            moe_impl=args.moe_impl,
        )
        self.model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
        self.model.load_tokenizer(args.model_id)
        self.image_processor = self.model.image_processor

        print("Loaded HunyuanImage3 pipeline")

    @staticmethod
    def standardize_message_list(message_list, context_mode="single_round"):
        processed_message_list = []

        # We always keep system message if available
        for message in message_list:
            if message["role"] == "system":
                processed_message_list.append(deepcopy(message))
            else:
                break
        if context_mode == "single_round":
            # Traverse the message list in reverse order to find all the last successive user messages.
            reversed_user_messages = []
            for message in reversed(message_list):
                if message["role"] == "user":
                    reversed_user_messages.append(deepcopy(message))
                else:
                    break
            processed_message_list.extend(reversed(reversed_user_messages))

        elif context_mode == "unlimited":
            processed_message_list = deepcopy(message_list)

        else:
            raise ValueError(f"Unknown message strategy: {context_mode}")
        return processed_message_list

    def _run_generation_in_thread(self, model_inputs, kwargs):
        """Run model generation in a thread with inference mode enabled."""
        with torch.inference_mode():
            self.model._generate(**model_inputs, **kwargs)  # noqa

    @torch.no_grad()
    def _generate(
            self,
            message_list: List[Dict[str, Any]],
            seed: Optional[int] = None,
            image_size: str = "auto",
            verbose: int = 1,
            **kwargs,
    ):
        """
        A uniform interface for all the t2i, general editing, lm, and mmu tasks.
        Only batch_size 1 is supported.

        Args:
            message_list (List[Dict[str, Any]]):
                A list of dictionaries containing the history messages and new questions.
                [
                    dict(role='system', type='text', content='xxxx', content_type='str')
                    dict(role='user', type='text', content='xxxx', content_type='str'),
                    dict(role='user', type='joint_image', content='xxxx', content_type='image_info'),
                    dict(role='assistant', type='text', content='xxxx', content_type='str')
                    dict(role='assistant', type='joint_image', content='xxxx', content_type='image_info')
                ]
            seed (Optional[int]):
                The random seed for deterministic results.
            image_size (str):
                The size of the generated images, can be "auto" or specified size.
            verbose (int):
                The verbosity level. 0 for silent, 1 for detailed info.
            kwargs:
                context_mode (str):
                    The context mode for processing the message_list, can be "single_round" or "unlimited".
                bot_task (str):
                    The task for the model, can be "image", "think", "recaption", or "auto".
                    "image": text-to-image generation, maybe predict image size first if image_size="auto".
                    "think": chain-of-thought text-to-image generation, predict image size first if image_size="auto".
                    "recaption": image editing with new caption, maybe predict image size first if image_size="auto".
                    "auto": text generation.
                drop_think (bool):
                    Whether to drop the <think> part in the context when generating image.
        """
        streamer = None
        model_inputs = None
        outputs = None

        try:
            try:
                context_mode = kwargs.pop("context_mode")
                message_list = self.standardize_message_list(message_list, context_mode=context_mode)
            except Exception as e:
                yield {"role": "assistant", "value": f"Error: {e}", "type": "text", "error": 100}
                return

            streamer = TextIteratorStreamer(self.model.tokenizer, skip_prompt=True, skip_special_tokens=False)
            bot_task = kwargs.get("bot_task")
            stop_token = ""
            bot_answer = ""

            # ================================================================
            # gen_text: plain text
            if bot_task != "image":
                model_inputs = self.model.prepare_model_inputs(
                    message_list=message_list, seed=seed, image_size=image_size, **kwargs,
                )
                model_inputs.update({"streamer": streamer, "verbose": verbose})

                # Use wrapper method to ensure inference_mode in thread
                thread = Thread(
                    target=self._run_generation_in_thread,
                    args=(model_inputs, kwargs),
                )
                thread.start()

                # Start token will not be returned by streamer, so we add it here if needed
                if bot_task in ["think", "recaption"]:
                    bot_answer = f"<{bot_task}>"
                    yield {"role": "system", "value": f"<{bot_task}>", "type": "text"}
                else:
                    bot_answer = ""
                stop_token = None
                for text_token in streamer:
                    stop_token = text_token
                    print(text_token, end="", flush=True)
                    if text_token.startswith("<boi>") or text_token.startswith("<img"):
                        continue
                    bot_answer += text_token
                    yield dict(role="assistant", value=text_token, type="text")
                print()
                # Ensure the generation thread completes
                thread.join()

            if stop_token.endswith("<|endoftext|>"):
                return

            # ================================================================
            # There are two paths to this branch:
            #   Assistant: <think> -> </think><recaption>xxx</recaption>
            #   Assistant: <recaption> -> xxx</recaption>
            if stop_token.endswith("</recaption>"):
                message_list.append(dict(
                    role="assistant", type="text", content=bot_answer, content_type="text",     # cot_text
                ))
                # Switch system_prompt to `en_recaption` if needed
                if kwargs.get("drop_think") and message_list[0]["role"] == "system":
                    message_list[0]["content"] = t2i_system_prompts["en_recaption"][0]

            # ================================================================
            # gen_text: img_ratio
            if image_size == "auto":
                kwargs.update({"bot_task": "img_ratio"})
                model_inputs = self.model.prepare_model_inputs(
                    message_list=message_list, seed=seed, image_size=image_size, **kwargs,
                )
                model_inputs.update({"streamer": streamer, "verbose": verbose})

                # Use wrapper method to ensure inference_mode in thread
                thread = Thread(
                    target=self._run_generation_in_thread,
                    args=(model_inputs, kwargs),
                )
                thread.start()

                stop_token = None
                for text_token in streamer:
                    time.sleep(0.01)
                    stop_token = text_token
                    print(text_token, end="", flush=True)
                print()
                # Ensure the generation thread completes
                thread.join()

            # ================================================================
            # stop_token can be (1) <boi> (image_size!=auto, bot_task=auto)
            #                   (2) </recaption> (image_size!=auto, bot_task=think/recaption)
            #                   (3) <img_ratio_*> (image_size=auto)
            # gen_image
            yield dict(role="assistant", value="image", type="flag")
            if image_size == "auto":
                if matched := re.search(r"<img_ratio_\d+>$", stop_token):
                    gen_image_info = self.image_processor.build_image_info(matched.group())
                else:
                    # Failed to predict image ratio, use the default one
                    gen_image_info = self.image_processor.build_image_info("1024x1024")
            else:
                gen_image_info = self.image_processor.build_image_info(image_size)
            message_list.append(dict(
                role="assistant", type="gen_image", content=gen_image_info, content_type="image_info"))
            # Here we enter the gen_image mode. The kwargs `bot_task` won't take effect.
            model_inputs = self.model.prepare_model_inputs(
                message_list=message_list, mode="gen_image", seed=seed, image_size=image_size, **kwargs,
            )
            
            # Generate image with inference mode
            with torch.inference_mode():
                outputs = self.model._generate(**model_inputs, **kwargs, verbose=verbose)   # noqa
            
            yield dict(role="assistant", value=outputs[0], type="image")

        finally:
            # Clean up large objects and free GPU memory after each request
            del streamer, model_inputs, outputs
            clear_gpu_memory()

    def gradio_image_to_image_info(self, image: gradio.components.image.Image) -> ImageInfo:
        img_path = image.value["path"]
        pil_image = Image.open(img_path).convert("RGB")
        image_info = self.image_processor.preprocess(pil_image)
        return image_info

    def history2messages(self, history):
        message_list = []

        # System message should only appear at the beginning of the conversation.
        for msg in history:
            if msg["role"] == "system":
                message_list.append(dict(
                    role="system", type="text", content=msg["content"], content_type='str'
                ))
            else:
                break

        for msg in history:
            if msg["role"] == "system":
                # Ignore system message in the middle of the conversation.
                continue
            elif msg["role"] in ["user", "assistant"]:
                if isinstance(msg["content"], str):
                    message_list.append(dict(
                        role=msg["role"], type="text", content=msg["content"], content_type='str'
                    ))
                elif isinstance(msg["content"], gradio.components.image.Image):
                    message_list.append(dict(
                        role=msg["role"],
                        type="joint_image",
                        content=self.gradio_image_to_image_info(msg['content']),
                        content_type='image_info',
                    ))
                else:
                    raise NotImplementedError(f"Unsupported message type: {type(msg['content'])}")
            else:
                raise NotImplementedError(f"Unsupported role: {msg['role']}")

        # Make sure the last message is from user
        if len(message_list) == 0 or message_list[-1]["role"] != "user":
            raise ValueError("The last message must be from user")

        return message_list

    def generate(self, history, **kwargs):
        message_list = self.history2messages(history)
        # Feed the message_list to the model and yield stream results
        yield from self._generate(message_list, **kwargs)
