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


import pickle

from matplotlib import pyplot as plt
import numpy as np
import scipy
import scipy.spatial
import torch
import pandas as pd


class MaterialListData:
    def __init__(self, data, data_fields, log_e=False):
        assert data.ndim == 2 and data.shape[0] > data.shape[1]
        self.data_fields = data_fields
        self.x_min, self.x_max = data.min(axis=0), data.max(axis=0)
        self.data_norm = self.normalize(data)
        print("data min:", self.x_min, "data max:", self.x_max)
        self.kdtree = None

    def sample_material_cluster(self, rng: np.random.Generator):
        cluster = rng.choice(self.n_clusters)
        in_cluster = self.cluster_labels == cluster
        n_in_cluster = in_cluster.sum()
        idxs_in_cluster = np.where(in_cluster)[0]
        idx_in_cluster = rng.choice(n_in_cluster)
        the_idx = idxs_in_cluster[idx_in_cluster]
        # print('sampled mat idx', idx_in_cluster, 'in cluster, total idx:', the_idx)
        item_norm = self.data_norm[the_idx]
        item_unnorm = self.unnormalize(item_norm)
        return item_norm, item_unnorm

    def compute_chunks(self):
        self.nonempty_voxel_indices = []
        self.nonempty_voxel_contained_indices = []

        # hardcoded bounds (!)
        n_per_dim = 10
        Es = np.linspace(0, 500, num=n_per_dim + 1, endpoint=True)
        nus = np.linspace(0, 0.5, num=n_per_dim + 1, endpoint=True)
        rhos = np.linspace(0, 10, num=n_per_dim + 1, endpoint=True)
        chunk_bounds = np.stack(np.meshgrid(Es, nus, rhos, indexing="ij"), axis=-1)

        self.voxel_bounds = (chunk_bounds - self.x_min) / (
            self.x_max - self.x_min
        ) * 2 - 1
        self.voxel_indices_3d_flat = np.stack(
            np.meshgrid(*[np.arange(n_per_dim)] * 3, indexing="ij"), axis=-1
        ).reshape(-1, 3)
        self.n_voxel_per_dim = n_per_dim
        self.n_voxels = n_per_dim**3
        assert self.voxel_indices_3d_flat.shape[0] == self.n_voxels
        viz_nonempty_chunk_idx = []
        for i_voxel in range(self.n_voxels):
            this_index_3d = self.voxel_indices_3d_flat[i_voxel]
            this_bounds_lower, this_bounds_upper = (
                self.voxel_bounds[*this_index_3d],
                self.voxel_bounds[*(this_index_3d + 1)],
            )
            items_in_this = np.where(
                np.all(
                    (self.data_norm >= this_bounds_lower)
                    & (self.data_norm < this_bounds_upper),
                    axis=-1,
                )
            )[0]
            if len(items_in_this) > 0:
                self.nonempty_voxel_indices.append(i_voxel)
                self.nonempty_voxel_contained_indices.append(items_in_this)
                viz_nonempty_chunk_idx.append(np.ones(len(items_in_this)) * i_voxel)

        self.n_nonempty_voxels = len(self.nonempty_voxel_indices)
        print("obtained", self.n_nonempty_voxels, "nonempty voxels")

        viz_nonempty_chunk_idx = np.concatenate(viz_nonempty_chunk_idx)
        viz_nonempty_indices = np.concatenate(self.nonempty_voxel_contained_indices)
        viz_nonempty_mat = self.data_norm[viz_nonempty_indices]
        viz_nonempty_mat = self.unnormalize(viz_nonempty_mat)

        import plotting
        from plotting import DEFAULT_FIGWIDTH

        fig, ax = plt.subplots(
            subplot_kw={"projection": "3d"},
            figsize=(DEFAULT_FIGWIDTH / 2 * 1.2, DEFAULT_FIGWIDTH / 2),
        )
        ax.view_init(elev=30, azim=-130, roll=0)
        im = ax.scatter(
            viz_nonempty_mat[:, 0],
            viz_nonempty_mat[:, 1],
            viz_nonempty_mat[:, 2],
            c=viz_nonempty_chunk_idx,
            marker="o",
            cmap="hsv",
            s=2,
        )
        # im.set_rasterized(True)
        ax.set_xlabel("$E / \\unit{\\GPa}$", labelpad=0), ax.set_ylabel(
            r"$\nu$", labelpad=1
        ), ax.set_zlabel(r"$\rho / (\unit{\gram\per\cubic\centi\metre})$", labelpad=0)
        plt.subplots_adjust(left=0.075, right=1.025, bottom=0.075, top=1.05)
        fig.savefig(
            f"material_chunks.pdf",
            dpi=500,
            pad_inches=0,
            bbox_inches=0,
        )
        plt.show()

    def match_to_voxels(self, points):
        self._ensure_kdtree()
        distances, indices = self.kdtree.query(points, k=1)
        matched_mats = self.data_norm[indices]
        assert matched_mats.shape == (len(points), 3)
        voxel_indices_3d = np.floor(
            (matched_mats + 1) / 2 / (1 + 1e-10) * self.n_voxel_per_dim
        )
        return voxel_indices_3d, matched_mats

    def sample_material_chunks(self, rng: np.random.Generator):
        nonempty_voxel_idx = rng.choice(self.n_nonempty_voxels)
        the_mat_idxs = self.nonempty_voxel_contained_indices[nonempty_voxel_idx]
        # print('voxel nonempty idx', nonempty_voxel_idx, 'total idx', self.nonempty_voxel_indices[nonempty_voxel_idx], 'contains', len(the_mat_idxs), 'mats.')
        idx_in_voxel = rng.choice(len(the_mat_idxs))
        the_idx = the_mat_idxs[idx_in_voxel]
        # print('  sampled mat idx', the_idx)
        item_norm = self.data_norm[the_idx]
        item_unnorm = self.unnormalize(item_norm)
        return item_norm, item_unnorm

    def sample_material_nochunks(self, rng: np.random.Generator):
        n_mats = self.data_norm.shape[0]
        the_idx = rng.integers(n_mats)
        item_norm = self.data_norm[the_idx]
        item_unnorm = self.unnormalize(item_norm)
        return item_norm, item_unnorm

    def _ensure_torch_cache(self, x):
        if not hasattr(self, "_x_minmax_torch"):
            self._x_minmax_torch = torch.from_numpy(
                np.stack([self.x_min, self.x_max], axis=0)
            ).to(x.dtype, x.device)

    def normalize(self, x):  # norms to [-1, 1]
        if isinstance(x, torch.Tensor):
            self._ensure_torch_cache(x)
            the_min, the_max = self._x_minmax_torch[0], self._x_minmax_torch[1]
        else:
            the_min, the_max = self.x_min, self.x_max
        return (x - the_min) / (the_max - the_min) * 2 - 1

    def unnormalize(self, x):
        if isinstance(x, torch.Tensor):
            self._ensure_torch_cache(x)
            the_min, the_max = self._x_minmax_torch[0], self._x_minmax_torch[1]
        else:
            the_min, the_max = self.x_min, self.x_max
        return (x + 1) / 2 * (the_max - the_min) + the_min

    def unnorm_scale(self, x):
        if isinstance(x, torch.Tensor):
            self._ensure_torch_cache(x)
            the_min, the_max = self._x_minmax_torch[0], self._x_minmax_torch[1]
        else:
            the_min, the_max = self.x_min, self.x_max
        return 0.5 * (the_max - the_min)

    def _ensure_kdtree(self):
        if not hasattr(self, "kdtree") or self.kdtree is None:
            self.kdtree = scipy.spatial.KDTree(self.data_norm)

    def query_distance(self, points):
        self._ensure_kdtree()
        distances, indices = self.kdtree.query(points, k=1)
        return distances


def compute_normalization():
    csv_file = "material_list.csv"
    target_name = "material_normalization"
    df = pd.read_csv(csv_file)
    assert not (df["nu"] >= 0.5).any()
    assert not (df["rho"] > 10).any()
    assert not (df["E"] > 500).any()
    print("# total", df.shape[0])  # 500 in paper

    ### Data processing (normalization) ###
    data = np.stack([df["E"], df["nu"], df["rho"]], axis=1)
    # E: Modulus of Elasticity (GPa)
    # nu: Poissons Ratio
    # rho: Density (g/cc)
    matlist_data = MaterialListData(
        data,
        ["Modulus of Elasticity (GPa)", "Poissons Ratio", "Density (g/cc)"],
        np.array([10**9, 1, 10**3]),
    )

    matlist_data.compute_chunks()

    # save normalization:
    with open(target_name + ".pkl", "wb") as f:
        pickle.dump(matlist_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    compute_normalization()
