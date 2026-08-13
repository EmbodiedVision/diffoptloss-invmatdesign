# Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
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
#
# This file is based on https://github.com/huggingface/diffusers/blob/3a28e36aa1ed7e62e70cdbe6166d90562fc22eb8/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
# and has been modified by Jens Kreber <jens.kreber@uni-a.de>.


from collections import defaultdict
import sys
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.pipelines import DDPMPipeline
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from ema_pytorch import EMA


# from diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py:59
def rescale_noise_cfg(noise_guided, noise_pred_cond, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_cond = noise_pred_cond.std(dim=list(range(1, noise_pred_cond.ndim)), keepdim=True)
    std_guided = noise_guided.std(dim=list(range(1, noise_guided.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_guided * (std_cond / std_guided)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_guided = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_guided
    return noise_guided


# based on diffusers/pipelines/ddpm/pipeline_ddpm.py
class NDConditionalDDPMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet: nn.Module, scheduler: DDPMScheduler, guidance_scale=1., guidance_rescale=0., ema_beta=0., ema_power=2/3):
        super().__init__()
        ema = EMA(unet, beta=ema_beta, power=ema_power, update_after_step=0, update_every=10)
        self.register_modules(unet=unet, scheduler=scheduler, ema=ema)
        self.unet: nn.Module = self.unet
        self.scheduler: DDPMScheduler = self.scheduler
        self.ema: EMA = self.ema
        self.guidance_scale = guidance_scale
        self.guidance_rescale = guidance_rescale
        self.set_progress_bar_config(disable=True) # disable tqdm during inference


    @torch.no_grad()
    def __call__(
        self,
        cond,
        batch_size: int|None = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        ds_indices = None,
        return_dict: bool = True,
        return_n_intermediate = 0,
        pred_orig_as_intermediate = False,
        override_params = {},
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        """
        num_inference_steps = override_params.get('num_inference_steps', num_inference_steps)
        conditional = cond is not None
        assert not (batch_size is None and not conditional), f"No way to dermine sample number."
        if batch_size is None:
            batch_size = cond.shape[0]
        elif conditional:
            assert cond.shape[0] == batch_size, f"cond shape is {cond.shape}, but batch_size is {batch_size}"
        
        spatial_shape = self.unet.config.sample_size
        shape = (batch_size, self.unet.config.channels,) + spatial_shape

        intermediate_samples = []
        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        if return_n_intermediate > 0:
            delta = num_inference_steps // (return_n_intermediate + 1)
            intermediate_timesteps = self.scheduler.timesteps.cpu().numpy()[delta::delta]
        else:
            intermediate_timesteps = []


        def model_fn(sample, cond, t):
            self.ema.eval()
            # classifier-free guidance
            # from diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py:1010
            # 1. predict noise model_output
            pred_cond = self.ema(sample, cond, t, drop_cond_prob=0.).sample

            if self.guidance_scale != 1:
                assert self.guidance_scale > 0
                if t == 999:
                    pred = pred_cond
                else:
                    assert self.scheduler.config.prediction_type == "v"
                    pred_uncond = self.ema(sample, cond, t, drop_cond_prob=1.).sample
                    eps_cond = self.scheduler.v_to_eps(pred_cond, sample, t)
                    eps_uncond = self.scheduler.v_to_eps(pred_uncond, sample, t)
                    eps_cond = eps_uncond + (- self.guidance_scale) * (eps_cond - eps_uncond)
                    pred = self.scheduler.eps_to_v(eps_cond, sample, t)
            else:
                pred = pred_cond
            if self.guidance_rescale > 0.:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                pred = rescale_noise_cfg(pred, pred_cond, guidance_rescale=self.guidance_rescale)
            return pred

        stats = defaultdict(list)

        sample = randn_tensor(shape, generator=generator, device=self.device)

        for i, t in enumerate(self.progress_bar(self.scheduler.timesteps)):
            # 2. compute previous data point: x_t -> x_t-1
            res = self.scheduler.step(None, t, sample, generator=generator, 
                                        ds_indices=ds_indices, cond=cond, override_params=override_params, model_fn=model_fn, i=i)
            sample = res.prev_sample
            assert not sample.requires_grad # assume the scheduler detaches
            for k, v in res.stats.items():
                stats[k].append(v)

            if t in intermediate_timesteps:
                intermediate_samples.append(res.pred_original_sample if pred_orig_as_intermediate else sample)

        for k in stats:
            if k.endswith('_sum'):
                stats[k] = sum(stats[k])

        if not return_dict:
            return (sample, intermediate_samples, intermediate_timesteps, stats)

        return ImagePipelineOutput(images=sample)
