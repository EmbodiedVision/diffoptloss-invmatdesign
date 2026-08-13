# Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
# Copyright 2024 UC Berkeley Team and The HuggingFace Team. All rights reserved.
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
#
# This file is based on https://github.com/huggingface/diffusers/blob/3a28e36aa1ed7e62e70cdbe6166d90562fc22eb8/src/diffusers/schedulers/scheduling_ddpm.py
# and has been modified by Jens Kreber <jens.kreber@uni-a.de>.

# DISCLAIMER: This file is strongly influenced by https://github.com/ermongroup/ddim


from collections import defaultdict
import math
from dataclasses import dataclass
import sys
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple, Union

import numpy as np
from scipy import sparse
import torch
from scipy.optimize import minimize

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
)


from dataset_utils import flatten_nodes, structure_nodes
from fem3d import Config3D as Config
import solver_interface
from spherical_composite_dataset import check_mat_plausibility


@dataclass
class DDPMSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None
    stats: Optional[dict] = None


def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


# Copied from diffusers.schedulers.scheduling_ddim.rescale_zero_terminal_snr
def rescale_zero_terminal_snr(betas):
    """
    Rescales betas to have zero terminal SNR Based on https://arxiv.org/pdf/2305.08891.pdf (Algorithm 1)


    Args:
        betas (`torch.Tensor`):
            the betas that the scheduler is being initialized with.

    Returns:
        `torch.Tensor`: rescaled betas with zero terminal SNR
    """
    # Convert betas to alphas_bar_sqrt
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_bar_sqrt = alphas_cumprod.sqrt()

    # Store old values.
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()

    # Shift so the last timestep is zero.
    alphas_bar_sqrt -= alphas_bar_sqrt_T

    # Scale so the first timestep is back to the old value.
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)

    # Convert alphas_bar_sqrt to betas
    alphas_bar = alphas_bar_sqrt**2  # Revert sqrt
    alphas = alphas_bar[1:] / alphas_bar[:-1]  # Revert cumprod
    alphas = torch.cat([alphas_bar[0:1], alphas])
    betas = 1 - alphas

    return betas



class GuidanceDDPMScheduler(SchedulerMixin, ConfigMixin):
    """
    `DDPMScheduler` explores the connections between denoising score matching and Langevin dynamics sampling.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.0001):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.02):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear`, `scaled_linear`, or `squaredcos_cap_v2`.
        trained_betas (`np.ndarray`, *optional*):
            An array of betas to pass directly to the constructor without using `beta_start` and `beta_end`.
        variance_type (`str`, defaults to `"fixed_small"`):
            Clip the variance when adding noise to the denoised sample. Choose from `fixed_small`, `fixed_small_log`,
            `fixed_large`, `fixed_large_log`, `learned` or `learned_range`.
        clip_sample (`bool`, defaults to `True`):
            Clip the predicted sample for numerical stability.
        clip_sample_range (`float`, defaults to 1.0):
            The maximum magnitude for sample clipping. Valid only when `clip_sample=True`.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True`.
        timestep_spacing (`str`, defaults to `"leading"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        rescale_betas_zero_snr (`bool`, defaults to `False`):
            Whether to rescale the betas to have zero terminal SNR. This enables the model to generate very bright and
            dark samples instead of limiting it to samples with medium brightness. Loosely related to
            [`--offset_noise`](https://github.com/huggingface/diffusers/blob/74fd735eb073eb1d774b1ab4154a0876eb82f055/examples/dreambooth/train_dreambooth.py#L506).
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1
    ignore_for_config = ["ds", "stats"]

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        variance_type: str = "fixed_small",
        clip_sample: bool = True,
        clip_sample_min=0.0,
        clip_sample_max=1.0,
        prediction_type: str = "epsilon",
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        timestep_spacing: str = "leading",
        steps_offset: int = 0,
        rescale_betas_zero_snr: bool = False,
        ds=None,
        guidance_config=None,
    ):
        if trained_betas is not None:
            self.betas = torch.tensor(trained_betas, dtype=torch.float32)
        elif beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_train_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                torch.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_train_timesteps,
                    dtype=torch.float32,
                )
                ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        elif beta_schedule == "sigmoid":
            # GeoDiff sigmoid schedule
            betas = torch.linspace(-6, 6, num_train_timesteps)
            self.betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(
                f"{beta_schedule} is not implemented for {self.__class__}"
            )

        # Rescale for zero SNR
        if rescale_betas_zero_snr:
            self.betas = rescale_zero_terminal_snr(self.betas)

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        # standard deviation of the initial noise distribution
        self.init_noise_sigma = 1.0

        # setable values
        self.custom_timesteps = False
        self.num_inference_steps = None
        self.timesteps = torch.from_numpy(
            np.arange(0, num_train_timesteps)[::-1].copy()
        )

        self.variance_type = variance_type
        self.ds = ds  # for all kinds of physics and correction stuff
        # self.stats = stats # for normalization

        self.grad_counter = 0
        self.cached_grad_x0 = None
        self.cached_grad_xt = None
        self.do_reset_solver = False

    def scale_model_input(
        self, sample: torch.Tensor, timestep: Optional[int] = None
    ) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError(
                "Can only pass one of `num_inference_steps` or `custom_timesteps`."
            )

        if timesteps is not None:
            for i in range(1, len(timesteps)):
                if timesteps[i] >= timesteps[i - 1]:
                    raise ValueError("`custom_timesteps` must be in descending order.")

            if timesteps[0] >= self.config.num_train_timesteps:
                raise ValueError(
                    f"`timesteps` must start before `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps}."
                )

            timesteps = np.array(timesteps, dtype=np.int64)
            self.custom_timesteps = True
        else:
            if num_inference_steps > self.config.num_train_timesteps:
                raise ValueError(
                    f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                    f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                    f" maximal {self.config.num_train_timesteps} timesteps."
                )

            self.num_inference_steps = num_inference_steps
            self.custom_timesteps = False

            # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://arxiv.org/abs/2305.08891
            if self.config.timestep_spacing == "linspace":
                timesteps = (
                    np.linspace(
                        0, self.config.num_train_timesteps - 1, num_inference_steps
                    )
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
            elif self.config.timestep_spacing == "leading":
                step_ratio = self.config.num_train_timesteps // self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = (
                    (np.arange(0, num_inference_steps) * step_ratio)
                    .round()[::-1]
                    .copy()
                    .astype(np.int64)
                )
                timesteps += self.config.steps_offset
            elif self.config.timestep_spacing == "trailing":
                step_ratio = self.config.num_train_timesteps / self.num_inference_steps
                # creates integer timesteps by multiplying by ratio
                # casting to int to avoid issues when num_inference_step is power of 3
                timesteps = np.round(
                    np.arange(self.config.num_train_timesteps, 0, -step_ratio)
                ).astype(np.int64)
                timesteps -= 1
            else:
                raise ValueError(
                    f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
                )

        self.timesteps = torch.from_numpy(timesteps).to(device)
        # Guidance
        guidance_params = self.config.guidance_config["guidance_params"]
        assert all(
            k
            in ["interval", "stop", "tfg_rho", "tfg_mu", "tfg_rho_sched", "tfg_mu_sched", "tfg_nrecur", "tfg_niter",
                "n_solver_restarts", "which_grad", "grad_max_mag",
            ]
            for k in guidance_params
        )

        guidance_interval = guidance_params.get("interval", 1)
        self.guidance_timesteps = self.timesteps[
            :-1:guidance_interval
        ]  # (!) exclude last timestep! This is how DPS does it.
        guidance_stop_t = guidance_params.get("stop", -1)
        self.guidance_timesteps = self.guidance_timesteps[
            self.guidance_timesteps >= guidance_stop_t
        ]
        n_timesteps = len(self.timesteps)
        n_solver_restarts = guidance_params.get("n_solver_restarts", 4)
        solver_restart_idxs = np.linspace(
            0, n_timesteps - 1, n_solver_restarts, endpoint=False
        ).astype(int)
        self.solver_restart_timesteps = self.timesteps[solver_restart_idxs]

        ## TFG schedulers
        # Note: timesteps decreasing!
        alphas_bar_used = self.alphas_cumprod[self.timesteps]
        dtype, device = self.alphas_cumprod.dtype, self.alphas_cumprod.device
        alphas_bar_prev = torch.concatenate(
            [
                torch.tensor(1.0, dtype=dtype, device=device)[None],
                self.alphas_cumprod[:-1],
            ]
        )
        alphas_bar_used_prev = alphas_bar_prev[self.timesteps]
        self.decreasing_scheduler = (
            1 - alphas_bar_used / alphas_bar_used_prev
        )  # das sind die beta_t
        self.decreasing_scheduler = (
            self.decreasing_scheduler
            / self.decreasing_scheduler.sum()
            * len(self.decreasing_scheduler)
        )
        self.increasing_scheduler = (
            alphas_bar_used / alphas_bar_used_prev
        )  # das sind die alpha_t ( = 1 - beta_t)
        self.increasing_scheduler = (
            self.increasing_scheduler
            / self.increasing_scheduler.sum()
            * len(self.increasing_scheduler)
        )
        self.increasing_scheduler_alt = self.decreasing_scheduler.flip(0)
        self.increasing_scheduler_lin = 2 * torch.linspace(
            0, 1, steps=len(self.timesteps), dtype=dtype, device=device
        )
        self.decreasing_scheduler_lin = self.increasing_scheduler_lin.flip(0)

    def get_current_beta_t(self, t):
        prev_t = self.previous_timestep(t)
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev
        return current_beta_t

    def _get_variance(self, t, predicted_variance=None, variance_type=None):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        if variance_type is None:
            variance_type = self.config.variance_type

        # hacks - were probably added for training stability
        if variance_type == "fixed_small":
            variance = variance
        # for rl-diffuser https://arxiv.org/abs/2205.09991
        elif variance_type == "fixed_small_log":
            variance = torch.log(variance)
            variance = torch.exp(0.5 * variance)
        elif variance_type == "fixed_large":
            variance = current_beta_t
        elif variance_type == "fixed_large_log":
            # Glide max_log
            variance = torch.log(current_beta_t)
        elif variance_type == "learned":
            return predicted_variance
        elif variance_type == "learned_range":
            min_log = torch.log(variance)
            max_log = torch.log(current_beta_t)
            frac = (predicted_variance + 1) / 2
            variance = frac * max_log + (1 - frac) * min_log

        return variance

    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = (
                sample.float()
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = (
            torch.clamp(sample, -s, s) / s
        )  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    def model_prediction(self, model_output, sample, t):
        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        if self.config.prediction_type == "epsilon":
            pred_original_sample = (
                sample - one_minus_alpha_bar_t ** (0.5) * model_output
            ) / alpha_bar_t ** (0.5)
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.config.prediction_type == "v":
            pred_original_sample = (alpha_bar_t**0.5) * sample - (
                one_minus_alpha_bar_t**0.5
            ) * model_output
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample` or"
                " `v_prediction`  for the DDPMScheduler."
            )
        return pred_original_sample

    def v_to_eps(self, v, sample, t):
        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        # v to eps
        clean = (alpha_bar_t**0.5) * sample - (one_minus_alpha_bar_t**0.5) * v
        eps = (sample - clean * alpha_bar_t ** (0.5)) / one_minus_alpha_bar_t ** (0.5)
        return eps

    def eps_to_v(self, eps, sample, t):
        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        # eps to v
        clean = (sample - one_minus_alpha_bar_t ** (0.5) * eps) / alpha_bar_t ** (0.5)
        v = ((alpha_bar_t**0.5) * sample - clean) / (one_minus_alpha_bar_t**0.5)
        return v

    def eps_to_clean(self, eps, sample, t):
        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        clean = (sample - one_minus_alpha_bar_t ** (0.5) * eps) / alpha_bar_t ** (0.5)
        return clean

    def clean_to_eps(self, clean, sample, t):
        alpha_bar_t = self.alphas_cumprod[t]
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        eps = (sample - clean * alpha_bar_t ** (0.5)) / one_minus_alpha_bar_t ** (0.5)
        return eps

    def step(
        self,
        model_output: Optional[torch.Tensor],
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
        ds_indices=None,
        cond=None,
        override_params={},
        model_fn=None,
        i=None,
    ) -> Union[DDPMSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep

        prev_t = self.previous_timestep(t)

        shape = sample.shape
        bs, *d_ = shape
        d = np.prod(d_)
        sq_d = np.sqrt(d)
        device, dtype = sample.device, sample.dtype

        # 1. compute alphas, betas
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        one_minus_alpha_bar_t = 1 - alpha_bar_t
        one_minus_alpha_bar_t_prev = 1 - alpha_bar_t_prev
        alpha_t = alpha_bar_t / alpha_bar_t_prev
        beta_t = 1 - alpha_t

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (
            alpha_bar_t_prev ** (0.5) * beta_t
        ) / one_minus_alpha_bar_t
        current_sample_coeff = (
            alpha_t ** (0.5) * one_minus_alpha_bar_t_prev / one_minus_alpha_bar_t
        )

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
        def get_pred_x0(model_output, current_sample, t):
            pred_original_sample_unmod = self.model_prediction(
                model_output, current_sample, t
            )
            pred_original_sample = pred_original_sample_unmod
            # 3. Clip or threshold "predicted x_0"
            if self.config.thresholding:
                pred_original_sample = self._threshold_sample(pred_original_sample)
            elif self.config.clip_sample:
                pred_original_sample = pred_original_sample.clamp(
                    self.config.clip_sample_min, self.config.clip_sample_max
                )
            return pred_original_sample, pred_original_sample_unmod

        def get_prev_mean(pred_x0, current_sample):
            # 5. Compute predicted previous sample µ_t
            # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            pred_prev_mean = (
                pred_original_sample_coeff * pred_x0
                + current_sample_coeff * current_sample
            )
            # print('t=', t.item(), 'orig', pred_original_sample_coeff.item(), 'current', current_sample_coeff.item())
            return pred_prev_mean

        def get_random_noise():
            # 6. Calc noise
            if t > 0:
                std_noise = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
                noise_std = self._get_variance(t, predicted_variance=None)
                if self.variance_type == "fixed_small_log":
                    pass
                elif self.variance_type == "learned_range":
                    noise_std = torch.exp(0.5 * noise_std)
                else:
                    noise_std = noise_std**0.5
                random_noise = std_noise * noise_std
            else:
                noise_std, random_noise = 0.0, 0.0
            # print('t=', t.item(),  'noise std', noise_std)
            return noise_std, random_noise

        # 6. Guidance
        stats = defaultdict(float)
        stats["it_count_sum"] = 0
        stats["solver_calls_per_sum"] = 0
        guidance_params = self.config.guidance_config["guidance_params"]
        objective = self.config.guidance_config["objective"]

        if (
            objective != "none"
            and (t in self.guidance_timesteps)
            and (project_start := self.config.guidance_config["project_starting"]) >= 0
            and i >= project_start
            and (i - project_start)
            % (project_int := self.config.guidance_config["project_interval"])
            == 0
        ): # do projection!
            noise_std, random_noise = get_random_noise()
            pred_x0, pred_x0_unmod = get_pred_x0(model_fn(sample, cond, t), sample, t)
            pred_x0_np = pred_x0.cpu().numpy()
            pred_x0_np = np.clip(pred_x0_np, -1, 1)
            material_stuff, _ = check_mat_plausibility(pred_x0_np, ds=self.ds)
            proj_x0 = []
            for i_smpl, mat_stuff in enumerate(material_stuff):
                matched_mat_mat = mat_stuff["matched_mats"][0]
                matched_mat_part = mat_stuff["matched_mats"][1]
                particle_labels = mat_stuff["particle_labels"]
                proj_x0_ = np.where(
                    particle_labels[..., None], matched_mat_part, matched_mat_mat
                )  # w, h, 3 or d, w, h, 3
                if particle_labels.ndim == 2:
                    proj_x0_ = proj_x0_[None]  # 1, w, h, 3
                proj_x0_ = np.moveaxis(proj_x0_, -1, 0)
                proj_x0.append(proj_x0_)
            proj_x0 = np.stack(proj_x0)
            proj_x0_t = torch.from_numpy(proj_x0).to(pred_x0.device, pred_x0.dtype)
            pred_x0 = proj_x0_t

            pred_prev_mean = get_prev_mean(pred_x0, sample)
            pred_prev_sample = pred_prev_mean + random_noise
            pred_original_sample = pred_x0
        elif objective != "none" and t in self.guidance_timesteps: # do gradient guidance!
            n_grad_reuse = self.config.guidance_config["reuse_grad_count"]
            guidance_method = self.config.guidance_config["method"]
            assert guidance_method != "none", f"Need a guidance method!"

            
            grad_max_mag = guidance_params.get("grad_max_mag", 0)
            which_grad = guidance_params.get("which_grad", "model")
            assert which_grad == "model"

            if objective == "solver":
                if t in self.solver_restart_timesteps:
                    self.do_reset_solver = True

                def get_guide_grad(pred_x0, current_sample, to_x0_only=False):
                    if self.grad_counter % n_grad_reuse == 0:
                        # talk to solver
                        interfaces = solver_interface.INTERFACES
                        x = pred_x0.detach().cpu().numpy()
                        bs = x.shape[0]
                        x_pre, x_for_exp = interfaces[0].prepare_send(x)  # (bs,d,N)
                        bs, d, N = x_pre.shape
                        assert d == 3

                        keep_sol = not self.do_reset_solver
                        _time = time.time()
                        grads_raw, Ks, Js, it_counts = solver_interface.communicate(
                            x_pre,
                            self.config.guidance_config["aligned_solvers"],
                            keep_sol=keep_sol,
                        )
                        stats["t_communicate_sum"] += time.time() - _time
                        self.do_reset_solver = False

                        # density grads
                        densities_unnorm = x_pre[:, 2]
                        grads_raw = np.concatenate(
                            [
                                grads_raw,
                                densities_unnorm[:, None, :]
                                * self.config.guidance_config["density_loss_factor"],
                            ],
                            axis=1,
                        )

                        grads_in_normed = interfaces[0].process_grad(grads_raw)
                        grads_x0 = torch.from_numpy(grads_in_normed).to(pred_x0.device)
                        # grads_for_stats = (
                        #     grads_x0.swapaxes(0, 1).reshape(3, bs, -1).abs()
                        # )

                        if to_x0_only:
                            grads_xt = None
                        else:
                            _time = time.time()
                            grads_xt = torch.autograd.grad(
                                inputs=current_sample,
                                outputs=pred_x0,
                                grad_outputs=grads_x0,
                            )[0]
                            stats["t_gradmodel_sum"] += time.time() - _time

                        stats["J"] = Js
                        stats["K"] = Ks
                        stats["it_count_sum"] += it_counts.sum()
                        stats["solver_calls_per_sum"] += 1

                        if grad_max_mag > 0:
                            norm_grads_x0 = torch.linalg.norm(
                                grads_x0.reshape(bs, -1), ord=2, dim=1
                            )
                            grads_x0 *= torch.minimum(
                                torch.ones_like(grads_x0),
                                grad_max_mag
                                / norm_grads_x0[:, None, None, None, None],
                            )
                            if grads_xt is not None:
                                norm_grads_xt = torch.linalg.norm(
                                    grads_xt.reshape(bs, -1), ord=2, dim=1
                                )
                                grads_xt *= torch.minimum(
                                    torch.ones_like(grads_xt),
                                    grad_max_mag
                                    / norm_grads_xt[:, None, None, None, None],
                                )

                        self.cached_grad_x0 = grads_x0
                        self.cached_grad_xt = grads_xt
                        self.grad_counter = 1

                    else:
                        grads_x0 = self.cached_grad_x0
                        grads_xt = self.cached_grad_xt
                        self.grad_counter += 1
                    return grads_x0, grads_xt

            else:
                raise NotImplementedError

            # implements https://arxiv.org/abs/2409.15761 
            tfg_rho = guidance_params.get("tfg_rho", 1)
            tfg_rho = override_params.get("tfg_rho", tfg_rho)
            tfg_rho_sched = guidance_params.get("tfg_rho_sched", "constant")
            tfg_rho_sched = override_params.get("tfg_rho_sched", tfg_rho_sched)
            tfg_rho *= {
                "increase": self.increasing_scheduler[i],
                "decrease": self.decreasing_scheduler[i],
                "constant": 1.0,
                "increase_alt": self.increasing_scheduler_alt[i],
                "increase_lin": self.increasing_scheduler_lin[i],
                "decrease_lin": self.decreasing_scheduler_lin[i],
            }[tfg_rho_sched]
            tfg_mu = guidance_params.get("tfg_mu", 0)
            tfg_mu = override_params.get("tfg_mu", tfg_mu)
            tfg_mu_sched = guidance_params.get("tfg_mu_sched", "constant")
            tfg_mu_sched = override_params.get("tfg_mu_sched", tfg_mu_sched)
            tfg_mu *= {
                "increase": self.increasing_scheduler[i],
                "decrease": self.decreasing_scheduler[i],
                "constant": 1.0,
            }[tfg_mu_sched]
            tfg_nrecur = guidance_params.get("tfg_nrecur", 1)
            tfg_nrecur = override_params.get("tfg_nrecur", tfg_nrecur)
            tfg_niter = guidance_params.get("tfg_niter", 1)
            tfg_niter = override_params.get("tfg_niter", tfg_niter)

            for i_recur in range(tfg_nrecur):
                noise_std, random_noise = get_random_noise()

                if guidance_method == "tfg":
                    if tfg_rho != 0.0:
                        with torch.inference_mode(False):
                            sample_g = sample.clone().requires_grad_(True)
                            _time = time.time()
                            model_output = model_fn(sample_g, cond, t)
                            stats["t_forward_trace_sum"] += time.time() - _time
                            pred_x0, pred_x0_unmod = get_pred_x0(
                                model_output, sample_g, t
                            )
                        grad_x0, Delta_t = get_guide_grad(pred_x0, sample_g, False)
                        del sample_g
                        pred_x0 = pred_x0.detach()
                        pred_x0_unmod = pred_x0_unmod.detach()
                        Delta_t *= -tfg_rho
                    else:
                        pred_x0, pred_x0_unmod = get_pred_x0(
                            model_fn(sample, cond, t), sample, t
                        )
                        Delta_t = torch.zeros_like(sample)

                    if tfg_mu != 0.0:
                        with torch.inference_mode(False):
                            ugd_sample = pred_x0.clone().requires_grad_(True)
                            for i_iter in range(tfg_niter):
                                if tfg_rho != 0 and i_iter == 0:
                                    ugd_grad = grad_x0
                                else:
                                    ugd_grad, _ = get_guide_grad(ugd_sample, None, True)
                            ugd_sample = ugd_sample - tfg_mu * ugd_grad
                        ugd_sample = ugd_sample.detach()
                        Delta_0 = ugd_sample - pred_x0
                    else:
                        Delta_0 = torch.zeros_like(sample)
                    pred_prev_mean = get_prev_mean(pred_x0, sample)
                    pred_prev_sample = pred_prev_mean + random_noise

                    pred_prev_sample += (
                        Delta_t + Delta_0 * alpha_bar_t_prev**0.5
                    )

                    xt_mean = (alpha_bar_t / alpha_bar_t_prev) ** (
                        0.5
                    ) * pred_prev_sample
                    random_noise_fw = randn_tensor(
                        shape, generator=generator, device=device, dtype=dtype
                    )
                    sample = xt_mean + beta_t ** (0.5) * random_noise_fw

                    pred_original_sample = pred_x0

                else:
                    raise ValueError(f"Do not know about guidance method {guidance_method}")
                # END guidance

        else: # NO guidance
            noise_std, random_noise = get_random_noise()
            _time = time.time()
            pred_x0, pred_x0_unmod = get_pred_x0(model_fn(sample, cond, t), sample, t)
            stats["t_forward_sum"] += time.time() - _time
            pred_prev_mean = get_prev_mean(pred_x0, sample)
            pred_prev_sample = pred_prev_mean + random_noise
            pred_original_sample = pred_x0

        if not return_dict:
            return (pred_prev_sample, stats)

        return DDPMSchedulerOutput(
            prev_sample=pred_prev_sample,
            pred_original_sample=pred_original_sample,
            stats=stats,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )
        return noisy_samples

    def get_velocity(
        self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as sample
        self.alphas_cumprod = self.alphas_cumprod.to(device=sample.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=sample.dtype)
        timesteps = timesteps.to(sample.device)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
        return velocity

    def __len__(self):
        return self.config.num_train_timesteps

    def previous_timestep(self, timestep):
        if self.custom_timesteps:
            index = (self.timesteps == timestep).nonzero(as_tuple=True)[0][0]
            if index == self.timesteps.shape[0] - 1:
                prev_t = torch.tensor(-1)
            else:
                prev_t = self.timesteps[index + 1]
        else:
            num_inference_steps = (
                self.num_inference_steps
                if self.num_inference_steps
                else self.config.num_train_timesteps
            )
            prev_t = timestep - self.config.num_train_timesteps // num_inference_steps
            assert (
                prev_t in self.timesteps or prev_t < 0
            ), f"prev_t = {prev_t} not in self.timesteps for timestep {timestep}"

        return prev_t
