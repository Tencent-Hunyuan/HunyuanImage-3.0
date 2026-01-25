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
"""
Image Processor Module for HunyuanImage-3.0

This module provides image processing utilities for the HunyuanImage-3.0 model,
including VAE and ViT image encoding, resolution handling, and logits processing.
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Callable, Union, Any
import random
import math

import torch
from PIL import Image
from torchvision import transforms
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import load_image
from transformers.models.siglip2.image_processing_siglip2_fast import Siglip2ImageProcessorFast
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

from .tokenization_hunyuan_image_3 import ImageInfo, ImageTensor, CondImage, Resolution, ResolutionGroup

InputImage = Union[Image.Image, str]


class SliceVocabLogitsProcessor(LogitsProcessor):
    """
    [`LogitsProcessor`] that performs vocab slicing, i.e. restricting probabilities with in some range. This processor
    is often used in multimodal discrete LLMs, which ensure that we only sample within one modality

    Args:
        vocab_start (`int`): start of slice, default None meaning from 0
        vocab_end (`int`): end of slice, default None meaning to the end of list
        when start and end are all None, this processor does noting

    """

    def __init__(self, vocab_start: int = None, vocab_end: int = None, **kwargs):
        if vocab_start is not None and vocab_end is not None:
            assert vocab_start < vocab_end, f"Ensure vocab_start {vocab_start} < vocab_end {vocab_end}"
        self.vocab_start = vocab_start
        self.vocab_end = vocab_end
        self.other_slices = kwargs.get("other_slices", [])

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits by slicing the vocabulary range.
        
        Args:
            input_ids: Input token IDs (not used but required by interface)
            scores: Logits scores to process
            
        Returns:
            Processed scores with vocabulary slicing applied
        """
        # Extract scores within the specified vocabulary range
        scores_processed = scores[:, self.vocab_start: self.vocab_end]
        # Append additional slices if specified
        for other_slice in self.other_slices:
            scores_processed = torch.cat([scores_processed, scores[:, other_slice[0]: other_slice[1]]], dim=-1)
        return scores_processed

    def __repr__(self):
        return (
            f"SliceVocabLogitsWarper(vocab_start={self.vocab_start}, "
            f"vocab_end={self.vocab_end}, other_slices={self.other_slices})"
        )


def resize_and_crop(
    image: Image.Image,
    target_size: Tuple[int, int],
    resample=Image.Resampling.LANCZOS,
    crop_type='center',
    crop_coords=None
) -> Image.Image:
    """
    Resize and crop an image to a target size while maintaining aspect ratio.
    
    Args:
        image: Input PIL Image to process
        target_size: Target size as (width, height) tuple
        resample: Resampling filter to use (default: LANCZOS)
        crop_type: Crop strategy - 'resize', 'center', 'random', or 'fixed'
        crop_coords: Coordinates for fixed crop type as (left, top) tuple
        
    Returns:
        Processed PIL Image with target size
    """
    tw, th = target_size  # target width and height
    w, h = image.size  # original width and height

    tr = th / tw  # target aspect ratio
    r = h / w  # original aspect ratio

    if crop_type == "resize":
        resize_width = tw
        resize_height = th
        crop_top = 0
        crop_left = 0
        image = image.resize((resize_width, resize_height), resample=resample)
    else:
        # maintain the aspect ratio
        if r < tr:
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        if crop_type == 'center':
            crop_top = int(round((resize_height - th) / 2.0))
            crop_left = int(round((resize_width - tw) / 2.0))
        elif crop_type == 'random':
            crop_top = random.randint(0, resize_height - th)
            crop_left = random.randint(0, resize_width - tw)
        elif crop_type == 'fixed':
            assert crop_coords is not None, 'crop_coords should be provided when crop_type is fixed.'
            crop_left, crop_top = crop_coords
        else:
            raise ValueError(f'crop_type must be center, random or fixed, but got {crop_type}')

        image = image.resize((resize_width, resize_height), resample=resample)
        image = image.crop((crop_left, crop_top, crop_left + tw, crop_top + th))

    return image


@dataclass
class ResolutionGroupConfig:
    """
    Configuration for resolution group settings.
    
    Attributes:
        base_size: Base resolution size for the group
        step: Step size for resolution increments (optional)
        align: Alignment factor for resolution calculations (default: 16)
    """
    base_size: int = None
    step: Optional[int] = None
    align: int = 16

    def to_dict(self):
        """Convert configuration to dictionary format."""
        return asdict(self)


@dataclass
class VAEInfo:
    """
    Information container for VAE (Variational Autoencoder) image encoding.
    
    Attributes:
        encoder_type: Type identifier for the VAE encoder
        down_h_factor: Downsampling factor for height dimension
        down_w_factor: Downsampling factor for width dimension
        patch_size: Size of patches used in processing
        h_factor: Computed height factor (down_h_factor * patch_size)
        w_factor: Computed width factor (down_w_factor * patch_size)
        image_type: Type of image encoding (default: "vae")
    """
    encoder_type: str
    down_h_factor: int = -1
    down_w_factor: int = -1
    patch_size: int = 1
    h_factor: int = -1
    w_factor: int = -1
    image_type: str = None

    def __post_init__(self):
        """Compute derived factors after initialization."""
        self.h_factor = self.down_h_factor * self.patch_size
        self.w_factor = self.down_w_factor * self.patch_size
        if self.image_type is None:
            self.image_type = "vae"


@dataclass
class ViTInfo:
    """
    Information container for ViT (Vision Transformer) image encoding.
    
    Attributes:
        encoder_type: Type identifier for the ViT encoder
        h_factor: Height factor for token calculation
        w_factor: Width factor for token calculation
        max_token_length: Maximum token length (pad to this length)
        processor: Image processor callable for ViT encoding
        image_type: Type of image encoding (derived from encoder_type if None)
    """
    encoder_type: str
    h_factor: int = -1
    w_factor: int = -1
    max_token_length: int = 0   # pad to max_token_length
    processor: Callable = field(default_factory=BaseImageProcessor)
    image_type: str = None

    def __post_init__(self):
        """Derive image type from encoder type if not specified."""
        if self.image_type is None:
            self.image_type = self.encoder_type.split("-")[0]


class HunyuanImage3ImageProcessor(object):
    """
    Main image processor for HunyuanImage-3.0 model.
    
    This processor handles image preprocessing for both VAE and ViT encoders,
    manages resolution groups, and provides utilities for image-to-tensor conversion
    and conditional image processing.
    """
    def __init__(self, config):
        """
        Initialize the image processor with configuration.
        
        Args:
            config: Configuration object containing model parameters
        """
        self.config = config

        # Initialize resolution group configuration
        self.reso_group_config = ResolutionGroupConfig(base_size=config.image_base_size)
        # Create VAE resolution group with standard and extra resolutions
        self.vae_reso_group = ResolutionGroup(
            **self.reso_group_config.to_dict(),
            extra_resolutions=[
                Resolution("1024x768"),
                Resolution("1280x720"),
                Resolution("768x1024"),
                Resolution("720x1280"),
            ]
        )
        # Logits processor for image ratio tokens (lazy initialization)
        self.img_ratio_slice_logits_processor = None
        # PIL image to tensor transformation pipeline
        self.pil_image_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # transform to [-1, 1]
        ])
        # Initialize VAE information
        self.vae_info = VAEInfo(
            encoder_type=config.vae_type,
            down_h_factor=config.vae_downsample_factor[0], down_w_factor=config.vae_downsample_factor[0],
            patch_size=config.patch_size,
        )

        # Initialize ViT processor based on encoder type
        if config.vit_type == "siglip2-so400m-patch16-naflex":
            self.vit_processor = Siglip2ImageProcessorFast.from_dict(config.vit_processor)
        else:
            raise ValueError(f"Unsupported vit_type: {config.vit_type}")
        # Initialize ViT information
        self.vit_info = ViTInfo(
            encoder_type=config.vit_type,
            h_factor=self.vit_processor.patch_size,
            w_factor=self.vit_processor.patch_size,
            max_token_length=self.vit_processor.max_num_patches,
            processor=self.vit_processor,
        )
        # Store attention and image type configurations
        self.cond_token_attn_type = config.cond_token_attn_type
        self.cond_image_type = config.cond_image_type

    def build_gen_image_info(self, image_size, add_guidance_token=False, add_timestep_r_token=False) -> ImageInfo:
        """
        Build image information for generation tasks.
        
        Args:
            image_size: Target image size in various formats (HxW, W:H, or <img_ratio_i>)
            add_guidance_token: Whether to add guidance token to the image info
            add_timestep_r_token: Whether to add timestep R token to the image info
            
        Returns:
            ImageInfo object containing image dimensions and token information
        """
        # Parse image size (HxW, H:W, or <img_ratio_i>)
        if isinstance(image_size, str):
            if image_size.startswith("<img_ratio_"):
                ratio_index = int(image_size.split("_")[-1].rstrip(">"))
                reso = self.vae_reso_group[ratio_index]
                image_size = reso.height, reso.width
            elif 'x' in image_size:
                image_size = [int(s) for s in image_size.split('x')]
            elif ':' in image_size:
                image_size = [int(s) for s in image_size.split(':')]
                assert len(image_size) == 2, f"`image_size` should be in the format of 'W:H', got {image_size}."
                # Note that ratio is width:height
                image_size = [image_size[1], image_size[0]]
            else:
                raise ValueError(
                    f"`image_size` should be in the format of 'HxW', 'W:H' or <img_ratio_i>, got {image_size}.")
            assert len(image_size) == 2, f"`image_size` should be in the format of 'HxW', got {image_size}."
        elif isinstance(image_size, (list, tuple)):
            assert len(image_size) == 2 and all(isinstance(s, int) for s in image_size), \
                f"`image_size` should be a tuple of two integers or a string in the format of 'HxW', got {image_size}."
        else:
            raise ValueError(f"`image_size` should be a tuple of two integers or a string in the format of 'WxH', "
                             f"got {image_size}.")
        # Get target size aligned to resolution group
        image_width, image_height = self.vae_reso_group.get_target_size(image_size[1], image_size[0])
        # Calculate token dimensions based on VAE factors
        token_height = image_height // self.vae_info.h_factor
        token_width = image_width // self.vae_info.w_factor
        # Get base size and ratio index for the target resolution
        base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(image_size[1], image_size[0])
        image_info = ImageInfo(
            image_type="gen_image", image_width=image_width, image_height=image_height,
            token_width=token_width, token_height=token_height, base_size=base_size, ratio_index=ratio_idx,
            add_guidance_token=add_guidance_token, add_timestep_r_token=add_timestep_r_token,
        )
        return image_info

    def as_image_tensor(self, image, image_type, **kwargs) -> ImageTensor:
        """
        Convert image to ImageTensor with appropriate metadata.
        
        Args:
            image: Input image (PIL Image or tensor)
            image_type: Type of image encoding ("vae", "siglip2", or "anyres")
            **kwargs: Additional arguments including origin_size and type-specific params
            
        Returns:
            ImageTensor with attached ImageInfo metadata
        """
        # Convert PIL Image to tensor if needed
        if isinstance(image, Image.Image):
            tensor = self.pil_image_to_tensor(image)
        else:
            tensor = image
        
        # Extract original image dimensions
        origin_size = kwargs["origin_size"]
        ori_image_width = origin_size[0]
        ori_image_height = origin_size[1]

        if image_type == "vae":
            # Validate tensor dimensions
            assert tensor.ndim == 3 or tensor.ndim == 4
            h, w = tensor.shape[-2], tensor.shape[-1]
            # Ensure dimensions are divisible by VAE factors
            assert (h % self.vae_info.h_factor == 0 and w % self.vae_info.w_factor == 0), \
                (f"Image size should be divisible by ({self.vae_info.h_factor}, {self.vae_info.w_factor}), "
                 f"but got ({h} x {w}).")
            # Calculate token dimensions
            tk_height = h // self.vae_info.h_factor
            tk_width = w // self.vae_info.w_factor
            # Get base size and ratio index
            base_size, ratio_idx = self.vae_reso_group.get_base_size_and_ratio_index(w, h)
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=w, image_height=h, token_width=tk_width, token_height=tk_height,
                base_size=base_size, ratio_index=ratio_idx,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vae_image"
        elif image_type == "siglip2":
            # Extract spatial shapes and attention mask for SigLIP2 encoding
            spatial_shapes = kwargs["spatial_shapes"]  # 2  (h, w)
            pixel_attention_mask = kwargs["pixel_attention_mask"]  # seq_len
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=spatial_shapes[1].item() * self.vit_info.w_factor,
                image_height=spatial_shapes[0].item() * self.vit_info.h_factor,
                token_width=spatial_shapes[1].item(),
                token_height=spatial_shapes[0].item(),
                image_token_length=self.vit_info.max_token_length,
                ori_image_width=ori_image_width,
                ori_image_height=ori_image_height,
            )
            tensor.section_type = "cond_vit_image"
            tensor.vision_encoder_kwargs = {
                "spatial_shapes": spatial_shapes,
                "pixel_attention_mask": pixel_attention_mask,
            }
        elif image_type == "anyres":
            token_width = kwargs["resized_image_width"] // self.vit_info.w_factor
            token_height = kwargs["resized_image_height"] // self.vit_info.h_factor
            tensor.i = ImageInfo(
                image_type=image_type,
                image_width=kwargs["resized_image_width"],
                image_height=kwargs["resized_image_height"],
                token_width=token_width,
                token_height=token_height,
                image_token_length=token_height * (token_width + 1) + 2,
            )
            tensor.section_type = "cond_vit_image"
        else:
            raise ValueError(f"Unknown image type: {image_type}")
        return tensor

    def vae_process_image(self, image, target_size, random_crop: bool | str = False) -> ImageTensor:
        """
        Process image for VAE encoding.
        
        Args:
            image: Input PIL Image
            target_size: Target size as (width, height) tuple
            random_crop: Crop strategy (bool or string: "center", "random", "resize")
            
        Returns:
            ImageTensor ready for VAE encoding
        """
        origin_size = image.size
        # Determine crop type from input parameter
        crop_type = random_crop if isinstance(random_crop, str) else ("random" if random_crop else "center")
        resized_image = resize_and_crop(image, target_size, crop_type=crop_type)
        return self.as_image_tensor(resized_image, image_type=self.vae_info.image_type, origin_size=origin_size)

    def vit_process_image(self, image) -> ImageTensor:
        """
        Process image for ViT encoding.
        
        Args:
            image: Input PIL Image
            
        Returns:
            ImageTensor ready for ViT encoding
        """
        origin_size = image.size
        # Process image through ViT processor
        inputs = self.vit_info.processor(image)
        image = inputs["pixel_values"].squeeze(0)   # (seq_len, dim)

        # Extract additional processor outputs (spatial shapes, attention masks, etc.)
        remain_keys = set(inputs.keys()) - {"pixel_values"}
        remain_kwargs = {}
        for key in remain_keys:
            if isinstance(inputs[key], torch.Tensor):
                remain_kwargs[key] = inputs[key].squeeze(0)
            else:
                remain_kwargs[key] = inputs[key]

        return self.as_image_tensor(
            image,
            image_type=self.vit_info.image_type,
            origin_size=origin_size,
            **remain_kwargs
        )

    def get_image_with_size(
            self,
            src: InputImage,
            random_crop: bool | str = False,
            return_type: str = "vae",
    ) -> tuple[ImageTensor | CondImage, bool]:
        """
        Process image for various generation tasks with dynamic image sizes.
        
        Args:
            src: Input image (PIL Image or file path)
            random_crop: Crop strategy for VAE processing
            return_type: Type of encoding to return ("vae", "vit", or "vae_vit")
            
        Returns:
            Tuple of (ImageTensor or CondImage, success_flag)
        """
        image = load_image(src)
        image_flag = "normal"
        img_success = image_flag != "gray"
        origin_size = image.size  # (w_ori, h_ori)

        # Process for VAE encoding if requested
        if "vae" in return_type:
            target_size = self.vae_reso_group.get_target_size(*origin_size)
            vae_image_tensor = self.vae_process_image(image, target_size, random_crop=random_crop)
        else:
            vae_image_tensor = None

        # Process for ViT encoding if requested
        if "vit" in return_type:
            vit_image_tensor = self.vit_process_image(image)
        else:
            vit_image_tensor = None

        if return_type == "vae":
            image_tensor = vae_image_tensor
        elif return_type == "vit":
            image_tensor = vit_image_tensor
        elif return_type == "vae_vit":
            image_tensor = CondImage(image_type=return_type, vae_image=vae_image_tensor, vit_image=vit_image_tensor)
        else:
            raise ValueError(f"Unknown return_type: {return_type}")

        return image_tensor, img_success

    def build_cond_images(
            self,
            image_list: Optional[list[InputImage]] = None,
            message_list: Optional[list[dict[str, Any]]] = None,
            infer_align_image_size: bool = False,
    ) -> Optional[list[CondImage]]:
        """
        Build conditional images from image list or message list.
        
        Args:
            image_list: Direct list of input images
            message_list: List of message dictionaries containing image information
            infer_align_image_size: Whether to align image sizes during processing
            
        Returns:
            List of CondImage objects or None if no images provided
        """
        if image_list is not None and message_list is not None:
            raise ValueError("`image_list` and `message_list` cannot be provided at the same time.")
        # Extract images from message list if provided
        if message_list is not None:
            image_list = []
            for message in message_list:
                # Find all image content in the message
                visuals = [
                    content
                    for content in message["content"]
                    if isinstance(content, dict) and content["type"] in ["image"]
                ]
                # Extract image sources from visual content
                image_list.extend([
                    vision_info[key]
                    for vision_info in visuals
                    for key in ["image", "url", "path", "base64"]
                    if key in vision_info and vision_info["type"] == "image"
                ])

        if infer_align_image_size:
            random_crop = "resize"
        else:
            random_crop = "center"

        return [
            self.get_image_with_size(src, return_type=self.cond_image_type, random_crop=random_crop)[0]
            for src in image_list
        ]

    def prepare_full_attn_slices(self, output, batch_idx=None, with_gen=True):
        """ Determine full attention image slices according to strategies. """
        if self.cond_image_type == "vae":
            cond_choices = dict(
                causal=[],
                full=output.vae_image_slices[batch_idx] if batch_idx is not None else output.vae_image_slices
            )

        elif self.cond_image_type == "vit":
            cond_choices = dict(
                causal=[],
                full=output.vit_image_slices[batch_idx] if batch_idx is not None else output.vit_image_slices
            )

        elif self.cond_image_type == "vae_vit":
            cond_choices = {
                "causal": [],
                "full": (
                    output.vae_image_slices[batch_idx] + output.vit_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.vae_image_slices + output.vit_image_slices
                ),
                "joint_full": (
                    output.joint_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.joint_image_slices
                ),
                "full_causal": (
                    output.vae_image_slices[batch_idx]
                    if batch_idx is not None
                    else output.vae_image_slices
                ),
            }

        else:
            raise ValueError(f"Unknown cond_image_type: {self.cond_image_type}")
        slices = cond_choices[self.cond_token_attn_type]

        if with_gen:
            gen_image_slices = (
                output.gen_image_slices[batch_idx]
                if batch_idx is not None
                else output.gen_image_slices
            )
            slices = slices + gen_image_slices
        return slices

    def build_img_ratio_slice_logits_proc(self, tokenizer):
        """
        Build logits processor for image ratio token slicing.
        
        This processor restricts token generation to image ratio tokens,
        ensuring the model only samples from the valid ratio token range.
        
        Args:
            tokenizer: Tokenizer instance with ratio token information
        """
        if self.img_ratio_slice_logits_processor is None:
            self.img_ratio_slice_logits_processor = LogitsProcessorList()
            # Add vocabulary slicer for ratio tokens
            self.img_ratio_slice_logits_processor.append(
                SliceVocabLogitsProcessor(
                    vocab_start=tokenizer.start_ratio_token_id,
                    vocab_end=tokenizer.end_ratio_token_id + 1,
                    other_slices=getattr(tokenizer, "ratio_token_other_slices", []),
                )
            )

    def postprocess_outputs(self, outputs: list[Image.Image], batch_cond_images, infer_align_image_size: bool = False):
        """
        Post-process generated images to align sizes with conditional images.
        
        Args:
            outputs: List of generated PIL Images
            batch_cond_images: List of conditional images for each output
            infer_align_image_size: Whether to align output image sizes to input sizes
            
        Returns:
            List of post-processed PIL Images
        """
        if infer_align_image_size:
            # Calculate target area based on base size
            target_area = self.vae_reso_group.base_size ** 2

            for batch_index, (output_image, cond_images) in enumerate(zip(outputs, batch_cond_images)):
                # Get ratio index for the output image
                output_image_ratio_index = self.vae_reso_group.get_base_size_and_ratio_index(
                    width=output_image.width, height=output_image.height
                )[1]
                # Collect ratio indices and original dimensions from conditional images
                cond_images_ratio_index_list = []
                cond_images_ori_width_list = []
                cond_images_ori_height_list = []
                for cond_image in cond_images:
                    if isinstance(cond_image, ImageTensor):
                        # Extract info from ImageTensor
                        cond_images_ratio_index_list.append(cond_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.i.ori_image_height)
                    else:  # CondImage
                        # Extract info from CondImage's VAE image
                        cond_images_ratio_index_list.append(cond_image.vae_image.i.ratio_index)
                        cond_images_ori_width_list.append(cond_image.vae_image.i.ori_image_width)
                        cond_images_ori_height_list.append(cond_image.vae_image.i.ori_image_height)

                if len(cond_images) == 0:
                    continue
                elif len(cond_images) == 1:
                    # Single conditional image: align if ratio matches
                    if output_image_ratio_index == cond_images_ratio_index_list[0]:
                        # Calculate ratio difference to check if alignment is needed
                        ratio_diff = abs(
                            cond_images_ori_height_list[0] / cond_images_ori_width_list[0] -
                            self.vae_reso_group[output_image_ratio_index].ratio
                        )
                        if ratio_diff >= 0.01:
                            # Calculate scaling factor to maintain area
                            scale = math.sqrt(
                                target_area / (cond_images_ori_width_list[0] * cond_images_ori_height_list[0])
                            )
                            new_w = round(cond_images_ori_width_list[0] * scale)
                            new_h = round(cond_images_ori_height_list[0] * scale)
                            # Resize output image to match conditional image dimensions
                            outputs[batch_index] = output_image.resize(
                                (new_w, new_h), resample=Image.Resampling.LANCZOS
                            )
                else:
                    # Multiple conditional images: align to first matching ratio
                    for cond_image_ratio_index, cond_image_ori_width, cond_image_ori_height in zip(
                        cond_images_ratio_index_list,
                        cond_images_ori_width_list,
                        cond_images_ori_height_list
                    ):
                        if output_image_ratio_index == cond_image_ratio_index:
                            # Calculate ratio difference for alignment check
                            ratio_diff = abs(
                                cond_image_ori_height / cond_image_ori_width -
                                self.vae_reso_group[output_image_ratio_index].ratio
                            )
                            if ratio_diff >= 0.01:
                                # Scale to match conditional image while preserving area
                                scale = math.sqrt(
                                    target_area / (cond_image_ori_width * cond_image_ori_height)
                                )
                                new_w = round(cond_image_ori_width * scale)
                                new_h = round(cond_image_ori_height * scale)
                                # Resize output to match conditional image
                                outputs[batch_index] = output_image.resize(
                                    (new_w, new_h), resample=Image.Resampling.LANCZOS
                                )
                            break

        return outputs

__all__ = [
    "HunyuanImage3ImageProcessor"
]
