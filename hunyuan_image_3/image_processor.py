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

from typing import Tuple

from PIL import Image
from torchvision import transforms
from transformers import Siglip2ImageProcessorFast

from .tokenizer_wrapper import ImageInfo, JointImageInfo, ResolutionGroup


def resize_and_crop(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
    tw, th = target_size
    w, h = image.size

    tr = th / tw
    r = h / w

    # resize
    if r < tr:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    image = image.resize((resize_width, resize_height), resample=Image.Resampling.LANCZOS)

    # center crop
    crop_top = int(round((resize_height - th) / 2.0))
    crop_left = int(round((resize_width - tw) / 2.0))

    image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))
    return image


class HunyuanImage3ImageProcessor(object):
    def __init__(self, config):
        self.config = config

        self.reso_group = ResolutionGroup(base_size=config.image_base_size)
        self.vae_processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
        ])
        self.vision_encoder_processor = Siglip2ImageProcessorFast.from_dict(config.vit_processor)

    def build_image_info(self, image_size):
        # parse image size (HxW, H:W, or <img_ratio_i>)
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif 'x' in image_size:
                image_size = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                image_size = [int(s) for s in image_size.split(':')]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'H:W' or <img_ratio_i>, got {image_size}.")
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {image_size}.")
        image_width, image_height = self.reso_group.get_target_size(image_size[1], image_size[0])
        token_height = image_height // (self.config.vae_downsample_factor[0] * self.config.patch_size)
        token_width = image_width // (self.config.vae_downsample_factor[1] * self.config.patch_size)
        base_size, ratio_idx = self.reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        image_info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, base_size=base_size, ratio_index=ratio_idx,
        )
        return image_info

    def preprocess(self, image: Image.Image):
        # ==== VAE processor ====
        image_width, image_height = self.reso_group.get_target_size(image.width, image.height)
        resized_image = resize_and_crop(image, (image_width, image_height))
        image_tensor = self.vae_processor(resized_image)
        token_height = image_height // (self.config.vae_downsample_factor[0] * self.config.patch_size)
        token_width = image_width // (self.config.vae_downsample_factor[1] * self.config.patch_size)
        base_size, ratio_index = self.reso_group.get_base_size_and_ratio_index(width=image_width, height=image_height)
        vae_image_info = ImageInfo(
            image_type="vae",
            image_tensor=image_tensor.unsqueeze(0),     # include batch dim
            image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height,
            base_size=base_size, ratio_index=ratio_index,
        )

        # ==== ViT processor ====
        inputs = self.vision_encoder_processor(image)
        image = inputs["pixel_values"].squeeze(0)  # seq_len x dim
        pixel_attention_mask = inputs["pixel_attention_mask"].squeeze(0)  # seq_len
        spatial_shapes = inputs["spatial_shapes"].squeeze(0)  # 2  (h, w)
        vision_encoder_kwargs = dict(
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        vision_image_info = ImageInfo(
            image_type="vit",
            image_tensor=image.unsqueeze(0),  # 1 x seq_len x dim
            image_width=spatial_shapes[1].item() * self.config.vit_processor["patch_size"],
            image_height=spatial_shapes[0].item() * self.config.vit_processor["patch_size"],
            token_width=spatial_shapes[1].item(),
            token_height=spatial_shapes[0].item(),
            image_token_length=self.config.vit_processor["max_num_patches"],
            # may not equal to token_width * token_height
        )
        return JointImageInfo(vae_image_info, vision_image_info, vision_encoder_kwargs)
