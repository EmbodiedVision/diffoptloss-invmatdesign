# Copyright 2026 University of Augsburg, Intelligent Perception in Technical Systems Group
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
# Code Author: Jens Kreber <jens.kreber@uni-a.de>


import math
import pickle
import sys, os
import glob

import numpy as np
import tqdm
import torch as th

from material_normalization import MaterialListData


class NotADict(dict):
    pass


def get_dataset(path, mat_norm_path=None):
    # old / simple case: single .npz file for everything
    if path.endswith(".npz"):
        with np.load(path) as f:
            d = { k: v for k, v in f.items() if k in [
                    "const_n_nodes_per_dim",
                    "const_elements",
                    "mat_ass",
                    "materials",
                    "AveK",
                    "particle_vol_frac",
                    "actual_vol_frac",
                    "circle_radius",
                ]
            }
    # new / complex case: const in .npz, dynamic in individual .npy files for mmap
    else:
        assert os.path.isdir(
            path
        ), f"Either specify a single .npz file or full path to a directory! Got {path}"
        d = NotADict()
        other_path = path + "/others.npz"
        if not os.path.exists(other_path):
            other_path = path + "/const.npz"
        with np.load(other_path) as f:
            for k, v in f.items():
                d[k] = v
        files = glob.glob(path + "/*.npy")
        for fn in files:
            k = os.path.basename(fn)[:-4]
            if k.startswith("_"):
                continue
            d[k] = np.lib.format.open_memmap(fn, mode="r")
    if mat_norm_path is not None:
        with open(mat_norm_path, "rb") as f:
            matlist_data: MaterialListData = pickle.load(f)
        d["matlist_data"] = matlist_data
    return d


def flatten_nodes(u, lib=np):
    transpose = {np: np.transpose, th: th.permute}[lib]
    assert u.ndim in [4, 5]  # [N,] (d, w, h, depth)
    has_batch_dim = u.ndim == 5
    u = u[None] if not has_batch_dim else u
    N, d, w, h, depth = u.shape
    n_nodes = w * h * depth
    u = transpose(u, (0, 1, -1, -2, -3)).reshape(N, d, n_nodes)
    u = u[0] if not has_batch_dim else u
    return u


def structure_nodes(u, n_nodes_per_dim, lib=np):
    transpose = {np: np.transpose, th: th.permute}[lib]
    assert u.ndim in [2, 3]  # [N,] (d, n_nodes)
    has_batch_dim = u.ndim == 3
    u = u[None] if not has_batch_dim else u
    N, d, n_nodes = u.shape
    assert n_nodes_per_dim[0] * n_nodes_per_dim[1] * n_nodes_per_dim[2] == n_nodes
    shape = (N, d, n_nodes_per_dim[0], n_nodes_per_dim[1], n_nodes_per_dim[2])
    trans_shape = (N, d, n_nodes_per_dim[2], n_nodes_per_dim[1], n_nodes_per_dim[0])
    u = transpose(u.reshape(trans_shape), (0, 1, -1, -2, -3))
    assert u.shape == shape
    u = u[0] if not has_batch_dim else u
    return u


def pad(x):
    # pad to nearest 2-exponential
    pads = [0, 0] + [
        (2 ** math.ceil(math.log(dim, 2)) - dim if dim > 1 else 0)
        for dim in x.shape[2:]
    ]
    x = np.pad(
        x,
        pad_width=tuple([(0, int(q)) for q in pads]),
        mode="constant",
        constant_values=0.0,
    )  # pad for binary shape # -> e.g. ((0,0), (0,0), (0,1), (0,1), (0,1))
    return x, pads


def unpad(x, pads):
    slices = [
        slice(-q) if q > 0 else slice(None) for q in pads
    ]  # if padding at position, take until last
    x = x[tuple(slices)]
    return x


def get_x_and_cond(ds, learn="mat"):
    N, _n_el = ds["mat_ass"].shape
    N_, n_mat, n_mat_components = ds["materials"].shape

    mat_per_el = ds["materials"][
        np.arange(N)[:, None], ds["mat_ass"][np.arange(N), :]
    ]  # (N, n_el, 3)

    if learn == "mat":
        n_el_per_dim = [q - 1 for q in ds["const_n_nodes_per_dim"]]
        ### normalize!
        matlist_data: MaterialListData = ds["matlist_data"]
        mat_per_el = matlist_data.normalize(mat_per_el)
        mat_per_el = mat_per_el.transpose((0, 2, 1))  # (N, n_el, 3) --> (N, 3, n_el)
        x = structure_nodes(mat_per_el, n_el_per_dim)  # --> (N, 3, n_x, n_y, n_z)
        x, padding = pad(x)
        # c: only for conditioning
        c_k = (
            ds["AveK"] / 700
        )  # min / max in default training set: 0.031, 671.7 -> 0, 700
        print("loaded mat data:", x.shape, "memory size:", x.nbytes / 10**6, "MB")
        c = np.concatenate([c_k], axis=1)
    else:
        raise ValueError(f"Cannot learn {learn}")

    return x, c, padding
