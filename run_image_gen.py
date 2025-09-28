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
from pathlib import Path
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from PE.deepseek import DeepSeekClient
from PE.system_prompt import system_prompt_universal, system_prompt_text_rendering


def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-system-prompt", type=str,
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using the custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Default to load from the model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt "
                                                          "is 'custom'.")
    parser.add_argument("--bot-task", type=str, choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
                             "Default to load from the model generation config.")
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--rewrite", default=True, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument("--sys-deepseek-prompt", type=str, choices=["universal", "text_rendering"], 
                        default="universal", help="System prompt for rewriting the prompt")

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)


def main(args):
    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="auto",
        moe_impl=args.moe_impl,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)
    model.load_tokenizer(args.model_id)

    if args.rewrite:
        # 通过环境变量获取DeepSeek的key_id和key_secret
        deepseek_key_id = os.getenv("DEEPSEEK_KEY_ID")
        deepseek_key_secret = os.getenv("DEEPSEEK_KEY_SECRET")
        if not deepseek_key_id or not deepseek_key_secret:
            raise ValueError(f"DeepSeek API key is not set!!! The Pretrain Checkpoint does not "
                             f"automatically rewrite or enhance input prompts, for optimal results currently,"
                             f"we recommend community partners to use deepseek to rewrite the prompts.")
        deepseek_client = DeepSeekClient(deepseek_key_id, deepseek_key_secret)
        
        if args.sys_deepseek_prompt == "universal":
            system_prompt = system_prompt_universal
        elif args.sys_deepseek_prompt == "text_rendering":
            system_prompt = system_prompt_text_rendering
        else:
            raise ValueError(f"Invalid system prompt: {args.sys_deepseek_prompt}")
        prompt, _ = deepseek_client.run_single_recaption(system_prompt, args.prompt)
        print("rewrite prompt: {}".format(prompt))
        args.prompt = prompt

    image = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        use_system_prompt=args.use_system_prompt,
        system_prompt=args.system_prompt,
        bot_task=args.bot_task,
        diff_infer_steps=args.diff_infer_steps,
        verbose=args.verbose,
        stream=True,
    )
   
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    image.save(args.save)
    print(f"Image saved to {args.save}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
