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
# This file is based on https://github.com/huggingface/diffusers/blob/3a28e36aa1ed7e62e70cdbe6166d90562fc22eb8/src/diffusers/models/unets/unet_1d.py
# and has been modified by Jens Kreber <jens.kreber@uni-a.de>.


import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import init

from .unet3d_blocks import get_down_block, get_up_block, UNetMidBlock3D

from einops import rearrange, reduce, repeat

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)

from positional_encodings.torch_encodings import PositionalEncodingPermute3D


@dataclass
class UNet3DOutput(BaseOutput):
    """
    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_channels, ...)`):
            The hidden states output from the last layer of the model.
    """

    sample: torch.Tensor


def prob_mask_like(shape, prob, device):  # from denoising diffusion
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


class MatUNet3DModel(ModelMixin, ConfigMixin):
    r"""
    A 3D UNet model that takes a noisy sample and a timestep and returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    @register_to_config
    def __init__(
        self,
        sample_size: int = None,
        sample_rate: Optional[int] = None,
        channels: int = 2,
        cond_channels: int = 0,
        cond_in_extra_net: bool = False,
        scalar_cond: int = 0,
        time_embedding_type: str = "fourier",
        flip_sin_to_cos: bool = True,
        freq_shift: float = 0.0,
        down_block_types: Tuple[str] = ("DownBlock3D", "AttnDownBlock3D"),
        up_block_types: Tuple[str] = ("AttnUpBlock3D", "UpBlock3D"),
        downsample_around=True,
        downsample_in_mid=False,
        downsample_type: str = "AvgPool",
        hidden_sizes=(16, 32),
        num_outer_layers: int = 2,
        mid_hidden_sizes=(64, 64),
        ignore_dims=(),
        act_fn: str = "swish",
        dropout: float = 0.0,
        attention_head_dim=8,
        norm_num_groups: int = 8,
        norm_eps=1e-5,
        final_residual=False,
        skip_connections=True,
        add_pe_for_attention=True,
        add_pe_elsewhere=False,
        extra_={},
    ):
        super().__init__()

        assert attention_head_dim is not None
        assert not add_pe_elsewhere
        self.extra = torch.nn.ModuleDict(extra_)

        self.sample_size = sample_size
        # at the very beginning, mapped by conv
        actual_in_channels = channels if cond_in_extra_net else channels + cond_channels
        cond_hidden_sizes = (
            tuple([q if q > 8 else 8 for i, q in enumerate(hidden_sizes)])
            if cond_in_extra_net
            else tuple([0 for _ in hidden_sizes])
        )
        kernel_size = [3, 3, 3]
        padding = [1, 1, 1]
        for d in ignore_dims:
            kernel_size[d] = 1
            padding[d] = 0
        kernel_size, padding = tuple(kernel_size), tuple(padding)

        if cond_in_extra_net:
            cond_channels_emb = (
                math.ceil(cond_channels / norm_num_groups) * norm_num_groups
            )
            self.cond_conv_in = nn.Conv3d(
                cond_channels, cond_channels_emb, kernel_size=1, padding=0
            )

        channels_emb = math.ceil(actual_in_channels / norm_num_groups) * norm_num_groups
        self.conv_in = nn.Conv3d(
            actual_in_channels, channels_emb, kernel_size=1, padding=0
        )

        ddd = 8
        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(
                embedding_size=ddd,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = 2 * ddd
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(
                ddd, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=freq_shift
            )
            timestep_input_dim = ddd
        time_embed_dim = ddd * 4

        if scalar_cond > 0:
            assert scalar_cond <= 2
            self.scalar_cond = scalar_cond
            self.scalar_cond_proj = GaussianFourierProjection(
                embedding_size=ddd,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            self.scalar_mlp = TimestepEmbedding(2 * ddd, time_embed_dim)
            if scalar_cond == 2:
                self.scalar_cond_proj_2 = GaussianFourierProjection(
                    embedding_size=ddd,
                    set_W_to_weight=False,
                    log=False,
                    flip_sin_to_cos=flip_sin_to_cos,
                )
                self.scalar_mlp_2 = TimestepEmbedding(2 * ddd, time_embed_dim)

            self.null_classes_emb = nn.Parameter(torch.empty(time_embed_dim))
            init.uniform_(self.null_classes_emb, -1, 1)  # fan in of 1
        else:
            # Note: not actively used
            self.null_classes_emb = nn.Parameter(torch.randn(cond_channels))

        self.time_mlp = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])
        self.out_block = None
        self.cond_down_blocks = nn.ModuleList([])

        # down
        down_input_channels = (channels_emb,) + hidden_sizes[:-1]
        down_output_channels = hidden_sizes
        if self.config.cond_in_extra_net:
            cond_down_input_channels = (cond_channels_emb,) + cond_hidden_sizes[:-1]
            cond_down_output_channels = cond_hidden_sizes
        else:
            cond_down_input_channels = tuple([0 for _ in cond_hidden_sizes])

        for i, down_block_type in enumerate(down_block_types):
            is_final_block = i == len(hidden_sizes) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=num_outer_layers,
                in_channels=down_input_channels[i],
                cond_channels=cond_down_input_channels[i],
                out_channels=down_output_channels[i],
                temb_channels=time_embed_dim,
                add_downsample=downsample_around,
                downsample_type=downsample_type,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim,
                kernel_size=kernel_size,
                padding=padding,
                resnet_time_scale_shift="default",
                dropout=dropout,
            )
            self.down_blocks.append(down_block)

            if cond_in_extra_net:
                cond_down_block = get_down_block(
                    down_block_type,
                    num_layers=num_outer_layers,
                    in_channels=cond_down_input_channels[i],
                    cond_channels=0,
                    out_channels=cond_down_output_channels[i],
                    temb_channels=time_embed_dim,
                    add_downsample=downsample_around,
                    downsample_type=downsample_type,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=attention_head_dim,
                    kernel_size=kernel_size,
                    padding=padding,
                    resnet_time_scale_shift="default",
                    dropout=dropout,
                )
                self.cond_down_blocks.append(cond_down_block)

        # mid
        self.mid_block = UNetMidBlock3D(
            hidden_sizes=mid_hidden_sizes,
            in_channels=hidden_sizes[-1],
            out_channels=hidden_sizes[-1],
            temb_channels=time_embed_dim,
            cond_channels=cond_hidden_sizes[-1],
            dropout=dropout,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=(
                attention_head_dim
                if attention_head_dim is not None
                else hidden_sizes[-1]
            ),
            resnet_groups=norm_num_groups,
            attn_groups=None,
            add_attention=True,
            add_pos_enc=add_pe_for_attention,
        )

        # up
        up_input_channels = list(reversed(hidden_sizes))
        up_skip_input_channels = (
            list(reversed(hidden_sizes))
            if skip_connections
            else [0] * len(hidden_sizes)
        )
        up_output_channels = list(reversed(hidden_sizes[:-1])) + [channels_emb]
        up_cond_channels = list(reversed(cond_down_input_channels))

        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(hidden_sizes) - 1

            up_block = get_up_block(
                up_block_type,
                num_layers=num_outer_layers,  # not doing the +1 for downsampling like in orig
                in_channels=up_input_channels[i],
                cond_channels=up_cond_channels[i],
                out_channels=up_output_channels[i],
                skip_in_channel=up_skip_input_channels[i],
                temb_channels=time_embed_dim,
                add_upsample=downsample_around,
                upsample_type=downsample_type,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=attention_head_dim,
                resnet_time_scale_shift="default",
                dropout=dropout,
                kernel_size=kernel_size,
                padding=padding,
            )
            self.up_blocks.append(up_block)

        self.conv_out = nn.Conv3d(channels_emb, channels, kernel_size=1, padding=0)

        self.final_residual = final_residual
        assert all(
            s == mid_hidden_sizes[0] for s in mid_hidden_sizes
        ), f"not all equal, need to implement that if you want it"
        self.pe = PositionalEncodingPermute3D(mid_hidden_sizes[-1])

        self.firstrun = False # whether to print architecture overview

    def forward(
        self,
        sample: torch.Tensor,
        cond,
        timestep: Union[torch.Tensor, float, int],
        return_dict: bool = True,
        drop_cond_prob=0.5,
    ) -> Union[UNet3DOutput, Tuple]:
        r"""
        The [`UNet3DModel`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch_size, num_channels, sample_size)`.
            timestep (`torch.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_3d.UNet3DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unets.unet_3d.UNet3DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_3d.UNet3DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(
                [timesteps], dtype=torch.long, device=sample.device
            )
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        timestep_enc = self.time_proj(timesteps)
        timestep_embed = self.time_mlp(timestep_enc)

        device = sample.device
        bs, c, w, h, dp = sample.shape
        orig_sample = sample

        # COND
        conditional = cond is not None
        if conditional:
            assert self.scalar_cond
            scalar_cond_proj = self.scalar_cond_proj(cond[:, 0])
            scalar_cond_emb = self.scalar_mlp(scalar_cond_proj)
            if self.scalar_cond == 2:
                scalar_cond_proj_2 = self.scalar_cond_proj_2(cond[:, 1])
                scalar_cond_emb_2 = self.scalar_mlp_2(scalar_cond_proj_2)
                scalar_cond_emb = scalar_cond_emb + scalar_cond_emb_2

            # derive condition, with condition dropout for classifier free guidance
            if drop_cond_prob > 0:
                keep_mask = prob_mask_like((bs,), 1 - drop_cond_prob, device=device)
                null_classes_emb = repeat(self.null_classes_emb, "d -> b d", b=bs)
                scalar_cond_emb = torch.where(
                    repeat(keep_mask, "b -> b d", d=self.null_classes_emb.shape[0]),
                    scalar_cond_emb,
                    null_classes_emb,
                )

            # just add to time!
            timestep_embed = timestep_embed + scalar_cond_emb
        else:
            assert (
                not self.config.cond_in_extra_net
            ), f"It makes no sense to have a cond net when there's no cond."

        if self.firstrun:
            print("Processing:")
            print("\tprepare input:", sample.shape, "->", end=" ")
        sample = self.conv_in(sample)
        if self.firstrun:
            print(sample.shape)

        # 2. down
        down_block_res_samples = ()

        for i, downsample_block in enumerate(self.down_blocks):
            cemb = cond_hidden if self.config.cond_in_extra_net else None
            if self.firstrun:
                print("\tdown", sample.shape, "+ cond", cemb.shape if cemb is not None else None)
            sample, res_samples = downsample_block(
                hidden_states=sample, temb=timestep_embed, pemb=None, cemb=cemb
            )
            down_block_res_samples += res_samples

            if self.config.cond_in_extra_net:
                cond_hidden, _ = self.cond_down_blocks[i](
                    cond_hidden, temb=timestep_embed
                )
                cond_samples += (cond_hidden,)
                cemb = (
                    cond_hidden  # only last one relevant for later since not overriden
                )

        if self.firstrun:
            print("\t ==>", sample.shape)

        # 3. mid
        if self.mid_block:
            if self.firstrun:
                print("\tmid ", sample.shape, "+ cond", cemb.shape if cemb is not None else None)
            sample = self.mid_block(sample, timestep_embed, cemb=cemb)
            if self.firstrun:
                print("\t ==>", sample.shape)

        # 4. up
        for i, upsample_block in enumerate(self.up_blocks):
            cemb = cond_samples[-i - 2] if self.config.cond_in_extra_net else None
            res_samples = (
                down_block_res_samples[-len(upsample_block.resnets) :]
                if self.config.skip_connections
                else None
            )
            down_block_res_samples = down_block_res_samples[
                : -len(upsample_block.resnets)
            ]
            if self.firstrun:
                print("\tup  ", sample.shape, "+ skip", [r.shape for r in res_samples] if res_samples is not None else None, "+ cond", cemb.shape if cemb is not None else None)
            sample = upsample_block(
                sample,
                res_hidden_states_tuple=res_samples,
                temb=timestep_embed,
                pemb=None,
                cemb=cemb,
            )
        if self.firstrun:
            print("\t ==>", sample.shape)

        # # 5. post-process
        if self.firstrun:
            print("final:", sample.shape, "->", end=" ")
        sample = self.conv_out(sample)
        if self.firstrun:
            print(sample.shape)
            self.firstrun = False

        if self.final_residual:
            sample += orig_sample

        if not return_dict:
            return (sample,)

        return UNet3DOutput(sample=sample)
