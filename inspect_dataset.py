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


import sys, os

import numpy as np
import matplotlib.pyplot as plt

import dataset_utils
from material_normalization import MaterialListData


def plot_k_histogram(Ks, percentiles, percentile_vals):
    import plotting
    from plotting import DEFAULT_FIGWIDTH, SMALL_SIZE

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(DEFAULT_FIGWIDTH / 2 * 0.98, DEFAULT_FIGWIDTH / 2 * 0.725),
        constrained_layout=True,
    )
    ax.set_xlim(-50, 450)
    counts, bins, patches = ax.hist(Ks, bins=50, histtype="stepfilled", alpha=0.9)
    maxcount = np.max(counts)
    ax.set_ylim(0, maxcount * 1.2)
    ax.set_xlabel("Bulk modulus $K / \\unit{\\GPa}$", labelpad=2)
    ax.set_ylabel("Count", labelpad=1)

    for i, (p, v) in enumerate(zip(percentiles, percentile_vals)):
        color = "tab:red"
        text = f"${p}\\%$"
        ypos_rel = 0.85
        ypos_data = maxcount * 1.05
        ax.axvline(
            v, 0, ypos_rel, linestyle="--", color=color, linewidth=1.0, alpha=0.8
        )

        label_v = v
        if i < 2:
            label_v += {0: -11, 1: +0}[i]

        ax.text(
            label_v,
            ypos_data,
            text,
            ha="center",
            va="bottom",
            fontsize=SMALL_SIZE,
            color=color,
        )

    fig.savefig(f"K_hist.pdf", dpi=800, pad_inches=0, bbox_inches=0)  # "tight",)


def plot_img(n_rows, pred_mat, titles=[], dimensionality=2, captions=None):
    plt.rcParams["image.cmap"] = "plasma"
    plt.rcParams["xtick.bottom"] = False
    plt.rcParams["xtick.labelbottom"] = False
    plt.rcParams["ytick.left"] = False
    plt.rcParams["ytick.labelleft"] = False
    n_slices = {1: NotImplemented, 2: 1, 3: 3}[dimensionality]

    n_cols = sum(
        [
            ((len(q) if isinstance(q, list) else 1) * n_slices if q is not None else 0)
            for q in [pred_mat]
        ]
    )
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(12, 15), constrained_layout=True, squeeze=False
    )
    col = 0

    if pred_mat is not None:
        if not isinstance(pred_mat, list):
            pred_mat = [pred_mat]
        for i, pred_mat_ in enumerate(pred_mat):
            assert pred_mat_.ndim == 5  # batch dim
            assert pred_mat_.shape[1] == 3  # 3 channels
            n_x = pred_mat_.shape[2]
            slices = [0] if n_x == 1 else [0, n_x // 4, n_x // 2]
            for i_x in slices:
                m = pred_mat_[:, :, i_x, :, :]  # use the channels, select x dim
                if i_x == slices[0]:
                    axes[0, col].set_title(
                        "Gen mat" + (" " + titles[i] if titles else ""), fontsize=8
                    )
                for r in range(n_rows):
                    toshow = (
                        m[r].transpose((1, 2, 0)) / 2 + 0.5
                    )  # channels to back for rgb
                    toshow[((toshow < 0) | (toshow > 1)).any(axis=-1)] = (0, 0, 0)
                    axes[r, col].imshow(toshow)
                    if captions is not None and captions[0] is not None:
                        val = captions[i][r]
                        if isinstance(val, str):
                            axes[r, col].set_xlabel(val)
                        else:
                            axes[r, col].set_xlabel(f"{val:e}")
                col += 1
    return fig


def plot_k_dists(Ks):
    import plotting
    from plotting import DEFAULT_FIGWIDTH

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(DEFAULT_FIGWIDTH / 2 * 0.98, DEFAULT_FIGWIDTH / 2 * 0.725),
        constrained_layout=True,
    )
    ax.set_xlim(-50, 450)
    counts, bins = np.histogram(Ks, bins=50)
    Ks_sorted = np.sort(Ks)
    Ks_diff = Ks_sorted[1:] - Ks_sorted[:-1]
    avg_diffs = []
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if i == len(bins) - 2:
            relevant_Ks = (Ks_sorted[:-1] >= left) & (Ks_sorted[:-1] <= right)
        else:
            relevant_Ks = (Ks_sorted[:-1] >= left) & (Ks_sorted[:-1] < right)
        this_diff_avg = Ks_diff[relevant_Ks].mean()
        avg_diffs.append(this_diff_avg)
    avg_diffs = np.stack(avg_diffs)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 25)
    ax.stairs(avg_diffs, bins, fill=True)
    ax.set_xlabel("Bulk modulus $K / \\unit{\\GPa}$", labelpad=2)
    ax.set_ylabel("$\\Delta K$ average over bin", labelpad=1)

    fig.savefig(f"K_delta.pdf", dpi=800, pad_inches=0, bbox_inches=0)


if __name__ == "__main__":
    try:
        path = sys.argv[1]
    except:
        print("usage:", sys.argv[0], "path_to_ds")
        sys.exit(1)

    ds = dataset_utils.get_dataset(path, mat_norm_path="material_normalization.pkl")
    matlist_data: MaterialListData = ds["matlist_data"]

    Ks = ds["AveK"][:, 0]  # remove trailing singleton dim
    percentiles = [1, 99]
    percentile_vals = np.percentile(Ks, percentiles)
    print("max K", np.max(Ks))
    print("K percentiles:", " ".join([f"{p}%" for p in percentiles]), percentile_vals)

    # PLOT K HISTOGRAM
    plot_k_histogram(Ks, percentiles, percentile_vals)

    # PLOT K AVG DIST
    plot_k_dists(Ks)

    plt.show(block=True)

    x, c, padding = dataset_utils.get_x_and_cond(ds, learn="mat")

    # Iterate through and visualize dataset
    BATCH_SIZE = 5
    for i in range(0, len(x), BATCH_SIZE):
        plot_img(BATCH_SIZE, pred_mat=x[i : i + BATCH_SIZE], dimensionality=2)

        plt.show(block=True)
