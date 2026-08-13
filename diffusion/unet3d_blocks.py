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
# This file is based on https://github.com/huggingface/diffusers/blob/3a28e36aa1ed7e62e70cdbe6166d90562fc22eb8/src/diffusers/models/unets/unet_3d_blocks.py
# and has been modified by Jens Kreber <jens.kreber@uni-a.de>.


import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, is_torch_version, logging
from diffusers.models.activations import get_activation
from .unet3d_components import ResnetBlock3D, Downsample3D, Upsample3D

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_down_block(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    cond_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    downsample_type: str,
    resnet_eps: float,
    resnet_act_fn: str,
    padding,
    kernel_size,
    resnet_groups: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
):
    down_block_type = (
        down_block_type[7:]
        if down_block_type.startswith("UNetRes")
        else down_block_type
    )
    downsample_conv = {"avgpool": False, "conv": True}[downsample_type.lower()]
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            cond_channels=cond_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            padding=padding,
            kernel_size=kernel_size,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            downsample_conv=downsample_conv,
        )


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        out_channels: int,
        temb_channels: int,
        padding: tuple[int],
        kernel_size: tuple[int],
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_conv=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            in_channels += cond_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    up=False,
                    down=False,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=downsample_conv,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        name="op",
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        cemb=None,
        pemb=None,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        output_states = ()

        if pemb is not None:
            hidden_states += pemb[None, : hidden_states.shape[1], :, :, :]

        for resnet in self.resnets:
            if cemb is not None:
                hidden_states = torch.concat([hidden_states, cemb], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, output_states


def get_up_block(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    cond_channels: int,
    out_channels: int,
    skip_in_channel: int,
    temb_channels: int,
    add_upsample: bool,
    upsample_type: str,
    resnet_eps: float,
    resnet_act_fn: str,
    padding,
    kernel_size,
    resolution_idx: Optional[int] = None,  # ?
    resnet_groups: Optional[int] = None,
    resnet_time_scale_shift: str = "default",
    attention_head_dim: Optional[int] = None,
    dropout: float = 0.0,
) -> nn.Module:

    up_block_type = (
        up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    )
    upsample_conv = {"avgpool": False, "conv": True}[upsample_type.lower()]
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            cond_channels=cond_channels,
            out_channels=out_channels,
            skip_in_channels=skip_in_channel,
            temb_channels=temb_channels,
            padding=padding,
            kernel_size=kernel_size,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            upsample_conv=upsample_conv,
        )


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        cond_channels: int,
        skip_in_channels: int,
        out_channels: int,
        temb_channels: int,
        padding: tuple[int],
        kernel_size: tuple[int],
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_conv=True,
    ):
        super().__init__()
        resnets = []

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        in_channels,
                        use_conv=upsample_conv,
                        kernel_size=kernel_size,
                        padding=padding,
                    )
                ]
            )
        else:
            self.upsamplers = None

        last_out = in_channels
        for i in range(num_layers):
            resnets.append(
                ResnetBlock3D(
                    in_channels=last_out + skip_in_channels + cond_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    up=False,
                    down=False,
                )
            )
            last_out = out_channels

        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False
        self.resolution_idx = resolution_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        res_hidden_states_tuple: Tuple[torch.Tensor, ...],
        temb: Optional[torch.Tensor] = None,
        upsample_size: Optional[int] = None,
        cemb=None,
        pemb=None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        is_freeu_enabled = (
            getattr(self, "s1", None)
            and getattr(self, "s2", None)
            and getattr(self, "b1", None)
            and getattr(self, "b2", None)
        )

        # modified to front
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        if pemb is not None:
            hidden_states += pemb[None, : hidden_states.shape[1], :, :, :]

        for resnet in self.resnets:
            # pop res hidden states
            if res_hidden_states_tuple is not None:
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            # FreeU: Only operate on the first two stages
            if is_freeu_enabled:
                raise RuntimeError
                hidden_states, res_hidden_states = apply_freeu(
                    self.resolution_idx,
                    hidden_states,
                    res_hidden_states,
                    s1=self.s1,
                    s2=self.s2,
                    b1=self.b1,
                    b2=self.b2,
                )

            if res_hidden_states_tuple is not None:
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            if cemb is not None:
                hidden_states = torch.cat([hidden_states, cemb], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet),
                        hidden_states,
                        temb,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb)

        return hidden_states


from diffusers.models.attention_processor import Attention
from positional_encodings.torch_encodings import PositionalEncodingPermute3D


class UNetMidBlock3D(nn.Module):
    """
    A 2D UNet mid-block [`UNetMidBlock2D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.Tensor`: The output of the last residual block, which is a tensor of shape `(batch_size, in_channels,
        height, width)`.

    """

    def __init__(
        self,
        hidden_sizes,  # e.g. (32, 32)
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        add_pos_enc: bool = True,
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        assert resnet_time_scale_shift == "default"
        if attn_groups is None:
            attn_groups = (
                resnet_groups if resnet_time_scale_shift == "default" else None
            )

        self.add_pos_enc = add_pos_enc
        if add_pos_enc:
            self.pos_enc = {}
            self.pos_enc_map = {}

        in_channels_ = (in_channels,) + hidden_sizes
        out_channels_ = hidden_sizes + (out_channels,)
        resnets = []
        attentions = []

        for i, (in_ch, out_ch) in enumerate(zip(in_channels_, out_channels_)):
            in_ch += cond_channels
            if i > 0 and self.add_attention:
                attentions.append(
                    Attention(
                        in_ch,
                        heads=in_ch // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
                if self.add_pos_enc:
                    if not in_ch in self.pos_enc:
                        self.pos_enc[str(in_ch)] = PositionalEncodingPermute3D(in_ch)
                    self.pos_enc_map[i] = str(in_ch)
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        if self.add_pos_enc:
            self.pos_enc = nn.ModuleDict(self.pos_enc)
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        cemb=None,
    ) -> torch.Tensor:
        if cemb is not None:
            hidden_states = torch.concat(
                [hidden_states, cemb], dim=1
            )  # add conditional
        hidden_states = self.resnets[0](hidden_states, temb)
        for i, (attn, resnet) in enumerate(zip(self.attentions, self.resnets[1:])):
            if cemb is not None:
                hidden_states = torch.concat(
                    [hidden_states, cemb], dim=1
                )  # add conditional
            if attn is not None:
                if self.add_pos_enc:
                    pe = self.pos_enc[self.pos_enc_map[i]]
                    hidden_states += pe(hidden_states)
                # fold into picture (transpose and stuff will be done in attn proc)
                orig_shape = hidden_states.shape
                hidden_states = hidden_states.reshape(orig_shape[:-2] + (-1,))
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = hidden_states.reshape(orig_shape)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states
