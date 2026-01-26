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
#
# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================================

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    r"""
    Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
    Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
    Flawed](https://arxiv.org/pdf/2305.08891.pdf).

    Args:
        noise_cfg (`torch.Tensor`):
            The predicted noise tensor for the guided diffusion process.
        noise_pred_text (`torch.Tensor`):
            The predicted noise tensor for the text-guided diffusion process.
        guidance_rescale (`float`, *optional*, defaults to 0.0):
            A rescale factor applied to the noise predictions.
    Returns:
        noise_cfg (`torch.Tensor`): The rescaled noise prediction tensor.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


@dataclass
class HunyuanImage3Text2ImagePipelineOutput(BaseOutput):
    samples: Union[List[Any], np.ndarray]


@dataclass
class FlowMatchDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
        reverse (`bool`, defaults to `True`):
            Whether to reverse the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            reverse: bool = True,
            solver: str = "euler",
            use_flux_shift: bool = False,
            flux_base_shift: float = 0.5,
            flux_max_shift: float = 1.15,
            n_tokens: Optional[int] = None,
    ):
        sigmas = torch.linspace(1, 0, num_train_timesteps + 1)

        if not reverse:
            sigmas = sigmas.flip(0)

        self.sigmas = sigmas
        # the value fed to model
        self.timesteps = (sigmas[:-1] * num_train_timesteps).to(dtype=torch.float32)
        self.timesteps_full = (sigmas * num_train_timesteps).to(dtype=torch.float32)

        self._step_index = None
        self._begin_index = None

        self.supported_solver = [
            "euler",
            "heun-2", "midpoint-2",
            "kutta-4",
        ]
        if solver not in self.supported_solver:
            raise ValueError(f"Solver {solver} not supported. Supported solvers: {self.supported_solver}")

        # empty dt and derivative (for heun)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    @property
    def state_in_first_order(self):
        return self.derivative_1 is None

    @property
    def state_in_second_order(self):
        return self.derivative_2 is None

    @property
    def state_in_third_order(self):
        return self.derivative_3 is None

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None,
                      n_tokens: int = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
        """
        self.num_inference_steps = num_inference_steps

        sigmas = torch.linspace(1, 0, num_inference_steps + 1)

        # Apply timestep shift
        if self.config.use_flux_shift:
            assert isinstance(n_tokens, int), "n_tokens should be provided for flux shift"
            mu = self.get_lin_function(y1=self.config.flux_base_shift, y2=self.config.flux_max_shift)(n_tokens)
            sigmas = self.flux_time_shift(mu, 1.0, sigmas)
        elif self.config.shift != 1.:
            sigmas = self.sd3_time_shift(sigmas)

        if not self.config.reverse:
            sigmas = 1 - sigmas

        self.sigmas = sigmas
        self.timesteps = (sigmas[:-1] * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)
        self.timesteps_full = (sigmas * self.config.num_train_timesteps).to(dtype=torch.float32, device=device)

        # empty dt and derivative (for kutta)
        self.derivative_1 = None
        self.derivative_2 = None
        self.derivative_3 = None
        self.dt = None

        # Reset step index
        self._step_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        return sample

    @staticmethod
    def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    @staticmethod
    def flux_time_shift(mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

    def sd3_time_shift(self, t: torch.Tensor):
        return (self.config.shift * t) / (1 + (self.config.shift - 1) * t)

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            pred_uncond: torch.FloatTensor = None,
            generator: Optional[torch.Generator] = None,
            n_tokens: Optional[int] = None,
            return_dict: bool = True,
    ) -> Union[FlowMatchDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            n_tokens (`int`, *optional*):
                Number of tokens in the input sequence.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
                isinstance(timestep, int)
                or isinstance(timestep, torch.IntTensor)
                or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        pred_uncond = pred_uncond.to(torch.float32) if pred_uncond is not None else None

        # dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]

        last_inner_step = True
        if self.config.solver == "euler":
            derivative, dt, sample, last_inner_step = self.first_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver in ["heun-2", "midpoint-2"]:
            derivative, dt, sample, last_inner_step = self.second_order_method(model_output, sigma, sigma_next, sample)
        elif self.config.solver == "kutta-4":
            derivative, dt, sample, last_inner_step = self.fourth_order_method(model_output, sigma, sigma_next, sample)
        else:
            raise ValueError(f"Solver {self.config.solver} not supported. Supported solvers: {self.supported_solver}")

        prev_sample = sample + derivative * dt

        # Cast sample back to model compatible dtype
        # prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        if last_inner_step:
            self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchDiscreteSchedulerOutput(prev_sample=prev_sample)

    def first_order_method(self, model_output, sigma, sigma_next, sample):
        derivative = model_output
        dt = sigma_next - sigma
        return derivative, dt, sample, True

    def second_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            # store for 2nd order step
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample

            derivative = model_output
            if self.config.solver == 'heun-2':
                dt = self.dt
            elif self.config.solver == 'midpoint-2':
                dt = self.dt / 2
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")
            last_inner_step = False

        else:
            if self.config.solver == 'heun-2':
                derivative = 0.5 * (self.derivative_1 + model_output)
            elif self.config.solver == 'midpoint-2':
                derivative = model_output
            else:
                raise NotImplementedError(f"Solver {self.config.solver} not supported.")

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def fourth_order_method(self, model_output, sigma, sigma_next, sample):
        if self.state_in_first_order:
            self.derivative_1 = model_output
            self.dt = sigma_next - sigma
            self.sample = sample
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_second_order:
            self.derivative_2 = model_output
            derivative = model_output
            dt = self.dt / 2
            last_inner_step = False

        elif self.state_in_third_order:
            self.derivative_3 = model_output
            derivative = model_output
            dt = self.dt
            last_inner_step = False

        else:
            derivative = (1/6 * self.derivative_1 + 1/3 * self.derivative_2 + 1/3 * self.derivative_3 +
                          1/6 * model_output)

            # 3. take prev timestep & sample
            dt = self.dt
            sample = self.sample
            last_inner_step = True

            # free dt and derivative
            # Note, this puts the scheduler in "first order mode"
            self.derivative_1 = None
            self.derivative_2 = None
            self.derivative_3 = None
            self.dt = None
            self.sample = None

        return derivative, dt, sample, last_inner_step

    def __len__(self):
        return self.config.num_train_timesteps


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
            self,
            pred_cond: torch.Tensor,
            pred_uncond: Optional[torch.Tensor],
            guidance_scale: float,
            step: int,
        ) -> torch.Tensor:

        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


class HunyuanImage3Text2ImagePipeline(DiffusionPipeline):
    r"""
    Pipeline for condition-to-sample generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        model ([`ModelMixin`]):
            A model to denoise the diffused latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `diffusion_model` to denoise the diffused latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    model_cpu_offload_seq = ""
    _optional_components = []
    _exclude_from_cpu_offload = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        model,
        scheduler: SchedulerMixin,
        vae,
        progress_bar_config: Dict[str, Any] = None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)
        # ==========================================================================================

        self.register_modules(
            model=model,
            scheduler=scheduler,
            vae=vae,
        )

        # should be a tuple or a list corresponding to the size of latents (batch_size, channel, *size)
        # if None, will be treated as a tuple of 1
        self.latent_scale_factor = self.model.config.vae_downsample_factor
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.latent_scale_factor)

        # Must start with APG_mode_
        self.cfg_operator = ClassifierFreeGuidance()

    @staticmethod
    def denormalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
        """
        Convert a PyTorch tensor to a NumPy image.
        """
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        return images

    @staticmethod
    def numpy_to_pil(images: np.ndarray):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_kwargs[k] = v
        return extra_kwargs

    def prepare_latents(self, batch_size, latent_channel, image_size, dtype, device, generator, latents=None):
        if self.latent_scale_factor is None:
            latent_scale_factor = (1,) * len(image_size)
        elif isinstance(self.latent_scale_factor, int):
            latent_scale_factor = (self.latent_scale_factor,) * len(image_size)
        elif isinstance(self.latent_scale_factor, tuple) or isinstance(self.latent_scale_factor, list):
            assert len(self.latent_scale_factor) == len(image_size), \
                "len(latent_scale_factor) shoudl be the same as len(image_size)"
            latent_scale_factor = self.latent_scale_factor
        else:
            raise ValueError(
                f"latent_scale_factor should be either None, int, tuple of int, or list of int, "
                f"but got {self.latent_scale_factor}"
            )

        latents_shape = (
            batch_size,
            latent_channel,
            *[int(s) // f for s, f in zip(image_size, latent_scale_factor)],
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(latents_shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    def set_scheduler(self, new_scheduler):
        self.register_modules(scheduler=new_scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int,
        image_size: List[int],
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_rescale: float = 0.0,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        model_kwargs: Dict[str, Any] = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The text to guide image generation.
            image_size (`Tuple[int]` or `List[int]`):
                The size (height, width) of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate samples closely linked to the
                `condition` at the expense of lower sample quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for sample
                generation. Can be used to tweak the same generation with different conditions. If not provided,
                a latents tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~DiffusionPipelineOutput`] instead of a
                plain tuple.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~DiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~DiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated samples.
        """

        callback_steps = kwargs.pop("callback_steps", None)
        pbar_steps = kwargs.pop("pbar_steps", None)

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale

        cfg_factor = 1 + self.do_classifier_free_guidance

        # Define call parameters
        device = self._execution_device

        # Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas,
        )

        # Prepare latent variables
        latents = self.prepare_latents(
            batch_size=batch_size,
            latent_channel=self.model.config.vae["latent_channels"],
            image_size=image_size,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
            latents=latents,
        )

        # Prepare extra step kwargs.
        _scheduler_step_extra_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator}
        )

        # Prepare model kwargs
        input_ids = model_kwargs.pop("input_ids")
        attention_mask = self.model._prepare_attention_mask_for_generation(     # noqa
            input_ids, self.model.generation_config, model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask.to(latents.device)

        # Sampling loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * cfg_factor)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                t_expand = t.repeat(latent_model_input.shape[0])

                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids,
                    images=latent_model_input,
                    timestep=t_expand,
                    **model_kwargs,
                )

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    model_output = self.model(**model_inputs, first_step=(i == 0))
                    pred = model_output["diffusion_prediction"]
                pred = pred.to(dtype=torch.float32)

                # perform guidance
                if self.do_classifier_free_guidance:
                    pred_cond, pred_uncond = pred.chunk(2)
                    pred = self.cfg_operator(pred_cond, pred_uncond, self.guidance_scale, step=i)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

                if i != len(timesteps) - 1:
                    model_kwargs = self.model._update_model_kwargs_for_generation(  # noqa
                        model_output,
                        model_kwargs,
                    )
                    if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                        input_ids = torch.gather(input_ids, 1, index=model_kwargs["position_ids"])

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor:
            latents = latents / self.vae.config.scaling_factor
        if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
            latents = latents + self.vae.config.shift_factor

        if hasattr(self.vae, "ffactor_temporal"):
            latents = latents.unsqueeze(2)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            image = self.vae.decode(latents, return_dict=False, generator=generator)[0]

        # b c t h w
        if hasattr(self.vae, "ffactor_temporal"):
            assert image.shape[2] == 1, "image should have shape [B, C, T, H, W] and T should be 1"
            image = image.squeeze(2)

        do_denormalize = [True] * image.shape[0]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if not return_dict:
            return (image,)

        return HunyuanImage3Text2ImagePipelineOutput(samples=image)
