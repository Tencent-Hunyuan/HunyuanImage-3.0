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
import time
from copy import deepcopy
from threading import Thread
from typing import List, Dict, Any, Optional

import gradio
import torch
from PIL import Image
from transformers import TextIteratorStreamer, LogitsProcessorList

from hunyuan_image_3.modeling_hunyuan_image_3 import HunyuanImage3ForCausalMM
from hunyuan_image_3.system_prompt import get_system_prompt


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
        A uniform interface for all t2i, editing, lm, and mmu tasks.
        Adapted for the Instruct model (modeling_hunyuan_image_3.py API).
        Only batch_size 1 is supported.
        """

        # Free any leftover state from previous generations
        gc.collect()
        torch.cuda.empty_cache()

        try:
            context_mode = kwargs.pop("context_mode")
            message_list = self.standardize_message_list(message_list, context_mode=context_mode)
        except Exception as e:
            yield {"role": "assistant", "value": f"Error: {e}", "type": "text", "error": 100}
            return

        bot_task = kwargs.pop("bot_task", "auto")
        drop_think = kwargs.pop("drop_think", False)
        model = self.model
        tkw = model._tokenizer
        image_processor = model.image_processor

        need_ratio = image_size == "auto" or bot_task == "img_ratio"
        cot_text = None
        batch_cond_images_cache = None

        # ==========================================================
        # bot_task == "auto": pure text generation, stream and return
        # ==========================================================
        if bot_task == "auto":
            streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=False)
            model_inputs = model.prepare_model_inputs(
                message_list=message_list, seed=seed, image_size=image_size,
                mode="gen_text", bot_task="auto", **kwargs,
            )
            model_inputs.update({"streamer": streamer, "verbose": verbose})

            thread = Thread(target=model.generate, kwargs=model_inputs)
            thread.start()

            eos = "<|endoftext|>"
            for text_token in streamer:
                print(text_token, end="", flush=True)
                if text_token in (eos, ""):
                    continue
                yield dict(role="assistant", value=text_token, type="text")
            print()
            thread.join()
            return

        # ==========================================================
        # bot_task in [think, recaption, think_recaption]: text gen phase with stage transitions
        # ==========================================================
        if bot_task in ("think", "recaption", "think_recaption"):
            first_bot_task = bot_task.split("_")[0]
            stage_transitions = []

            # think -> recaption transition
            if first_bot_task == "think" and "recaption" in bot_task:
                stage_transitions.append(
                    (tkw.end_of_think_token_id, [tkw.convert_tokens_to_ids(tkw.recaption_token)])
                )

            # ratio prediction transition
            if need_ratio:
                answer_prefix_tokens = []
                if getattr(model.generation_config, "sequence_template", "pretrain") == "instruct":
                    answer_prefix_tokens = [tkw.convert_tokens_to_ids(tkw.answer_token)]
                image_base_size = image_processor.vae_reso_group.base_size
                if "recaption" in bot_task:
                    transition_id = tkw.end_of_recaption_token_id
                else:
                    transition_id = tkw.end_of_think_token_id
                stage_transitions.append(
                    (transition_id, answer_prefix_tokens + [tkw.boi_token_id, tkw.size_token_id(image_base_size)])
                )
                final_stop_tokens = list(range(tkw.start_ratio_token_id, tkw.end_ratio_token_id + 1))
                for start, end in getattr(tkw, "ratio_token_other_slices", []):
                    final_stop_tokens.extend(range(start, end))
            else:
                if "recaption" in bot_task:
                    final_stop_tokens = [tkw.end_of_recaption_token_id]
                else:
                    final_stop_tokens = [tkw.end_of_think_token_id, tkw.end_of_recaption_token_id]

            # Build logits processor for ratio prediction
            logits_processor = None
            if need_ratio:
                image_base_size = image_processor.vae_reso_group.base_size
                logits_processor = LogitsProcessorList([
                    model._ConditionalSliceVocabLogitsProcessor(
                        trigger_token_ids=[tkw.size_token_id(image_base_size)],
                        vocab_start=tkw.start_ratio_token_id,
                        vocab_end=tkw.end_ratio_token_id + 1,
                        other_slices=getattr(tkw, "ratio_token_other_slices", []),
                        force_greedy=True,
                    )
                ])

            # Prepare model inputs for text gen phase
            model_inputs = model.prepare_model_inputs(
                message_list=message_list, seed=seed, max_new_tokens=2048,
                mode="gen_text", bot_task=first_bot_task, **kwargs,
            )
            batch_cond_images_cache = model_inputs['batch_cond_images']
            input_length = model_inputs["input_ids"].shape[1]

            # Stream text generation with stage transitions
            streamer = TextIteratorStreamer(model.tokenizer, skip_prompt=True, skip_special_tokens=False)
            model_inputs["streamer"] = streamer
            model_inputs["verbose"] = verbose

            gen_kwargs = dict(
                **model_inputs,
                decode_text=False,
                stage_transitions=stage_transitions if stage_transitions else None,
                final_stop_tokens=final_stop_tokens,
                logits_processor=logits_processor,
            )

            thread = Thread(target=model.generate, kwargs=gen_kwargs)
            thread.start()

            # Yield the opening tag for the first stage
            bot_answer = f"<{first_bot_task}>"
            yield {"role": "system", "value": f"<{first_bot_task}>", "type": "text"}

            for text_token in streamer:
                print(text_token, end="", flush=True)
                if text_token.startswith("<boi>") or text_token.startswith("<img"):
                    continue
                bot_answer += text_token
                yield dict(role="assistant", value=text_token, type="text")
            print()
            thread.join()

            if first_bot_task == "think":
                cot_text = [tkw.think_token + bot_answer.lstrip(f"<{first_bot_task}>")]
            else:
                cot_text = [tkw.recaption_token + bot_answer.lstrip(f"<{first_bot_task}>")]

            if drop_think and tkw.think_token in cot_text[0]:
                if tkw.recaption_token in cot_text[0]:
                    recaption_part = cot_text[0].split(tkw.recaption_token)[1]
                    if tkw.end_of_recaption_token in recaption_part:
                        recaption_part = recaption_part.split(tkw.end_of_recaption_token)[0]
                    cot_text = [tkw.recaption_token + recaption_part + tkw.end_of_recaption_token]

                    sys_msg = next((m for m in message_list if m["role"] == "system"), None)
                    if sys_msg:
                        sys_msg["content"] = get_system_prompt("en_recaption", bot_task) or ""

            # Free all text gen state before image gen
            del model_inputs, gen_kwargs, streamer, thread
            del stage_transitions, final_stop_tokens, logits_processor
            del batch_cond_images_cache
            batch_cond_images_cache = None
            gc.collect()
            torch.cuda.empty_cache()

        # ==========================================================
        # bot_task == "image": no text to stream
        # ==========================================================
        elif bot_task == "image":
            pass

        # ==========================================================
        # Image generation phase
        # ==========================================================
        yield dict(role="assistant", value="image", type="flag")

        _cot_text, outputs = model.generate_image(
            message_list=message_list, seed=seed, image_size=image_size,
            bot_task="image", cot_text=cot_text, **kwargs,
        )
        result_image = outputs[0]
        del _cot_text, outputs, cot_text
        gc.collect()
        torch.cuda.empty_cache()

        yield dict(role="assistant", value=result_image, type="image")

    def history2messages(self, history):
        """Convert Gradio chat history to OpenAI-style message list."""
        message_list = []

        # System messages first
        for msg in history:
            if msg["role"] == "system":
                message_list.append(dict(role="system", content=msg["content"]))
            else:
                break

        for msg in history:
            if msg["role"] == "system":
                continue
            elif msg["role"] in ["user", "assistant"]:
                if isinstance(msg["content"], str):
                    message_list.append(dict(role=msg["role"], content=msg["content"]))
                elif isinstance(msg["content"], gradio.components.image.Image):
                    img_path = msg["content"].value["path"]
                    pil_image = Image.open(img_path).convert("RGB")
                    message_list.append(dict(
                        role=msg["role"],
                        content=[{"type": "image", "image": pil_image}],
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
        try:
            yield from self._generate(message_list, **kwargs)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
