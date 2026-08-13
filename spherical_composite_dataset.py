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


import argparse
import concurrent.futures
from copy import deepcopy
import itertools
import pickle
import subprocess
import sys, os
from collections import defaultdict
import math
from tempfile import NamedTemporaryFile

import numpy as np
import tqdm


from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
import warnings
from skimage.morphology import skeletonize
from scipy import ndimage

from fem3d import read_in_file_3d, write_in_file_3d
from dataset_utils import structure_nodes, flatten_nodes, unpad
from material_normalization import MaterialListData


def fit_one(mati: np.ndarray, matlist_data: MaterialListData, fitting_seed: int = 0):
    assert mati.ndim == 4  # (d, n_x, n_y, n_z)
    d, n_x, n_y, n_z = mati.shape
    assert d == 3
    n_pxl = n_x * n_y * n_z
    is_2d = n_x == 1  # x = 1
    im_shp = mati.shape[2:] if is_2d else mati.shape[1:]
    variance_sum = 0
    # per material component
    labels_all = np.zeros(im_shp + (3,))
    fitted_mean_distances = [-1, -1, -1]
    fitted = True
    stats = {}

    gmm = GaussianMixture(
        n_components=2,
        max_iter=10,
        random_state=fitting_seed,
        init_params="k-means++",
        covariance_type="spherical",
    )  # todo: vary seed?
    mati_flat = mati.reshape(3, n_pxl).T  # (n_pxl, 3)
    try:
        gmm.fit(mati_flat)
    except ConvergenceWarning:
        print("ConvergenceWarning:", gmm.covariances_, "weights", gmm.weights_, "means", gmm.means_)
    except ValueError as e:
        print("fitting got ValueError: " + str(e) + " -- will continue with single component.")
        fitted = False
    if fitted:
        variance_sum = gmm.covariances_.sum().item()
        labels = gmm.predict(mati_flat).astype(bool)
        stats["fit_variance_sum"] = variance_sum
    else:
        variances = mati_flat.var(axis=0)
        print("fitting failed, variances:", variances)
        print(
            "\tmean",
            mati_flat.mean(axis=0),
            "min",
            mati_flat.min(axis=0),
            "max",
            mati_flat.max(axis=0),
        )
        print("\tnan?", np.isnan(mati_flat).any(), "inf?", np.isinf(mati_flat).any())
        variance_sum = variances.sum().item()
        labels = np.zeros_like(mati_flat[:, 0]).astype(bool)
        stats["fit_variance_sum"] = variance_sum

    labels = labels.reshape(im_shp)
    mati_im = mati_flat.reshape(im_shp + (3,))  # im_shp+(3,)
    n_a, n_b = (~labels).sum(), (labels).sum()
    do_a, do_b = n_a > 0, n_b > 0

    mask_variance_sum = 0
    if not (do_a and do_b):
        mat_matrix = mati_flat.mean(axis=0)  # (3,)
        mat_particles = mat_matrix
        mask_variance_sum = mati_flat.var(axis=0).sum()  # ()
        volume_fraction = 0
        circle_dia_frac = 0
        circle_dia_var = 0
        labels_return = np.zeros_like(labels)
    else:  # do both
        if is_2d:
            index_grid = np.stack(
                np.meshgrid(np.arange(n_y), np.arange(n_z), indexing="ij"), axis=-1
            )
        else:
            index_grid = np.stack(
                np.meshgrid(
                    np.arange(n_x), np.arange(n_y), np.arange(n_z), indexing="ij"
                ),
                axis=-1,
            )

        MIN_CIRC_DIA = 0.15
        MAX_CIRC_DIA = 0.4

        c_dia_medians = {False: None, True: None}
        c_dia_vars = {False: np.inf, True: np.inf}
        for invert in [False, True]:
            the_particle_labels = (~labels) if invert else labels
            if is_2d:
                n_active_border_points = (
                    the_particle_labels[0, :].sum()
                    + the_particle_labels[-1, :].sum()
                    + the_particle_labels[:, 0].sum()
                    + the_particle_labels[:, -1].sum()
                )
                n_total_border_points = 2 * n_y + 2 * n_z
            else:
                n_active_border_points = (
                    the_particle_labels[0, :, :].sum()
                    + the_particle_labels[-1, :, :].sum()
                    + the_particle_labels[:, 0, :].sum()
                    + the_particle_labels[:, -1, :].sum()
                    + the_particle_labels[:, :, 0].sum()
                    + the_particle_labels[:, :, -1].sum()
                )
                n_total_border_points = 2 * n_x * n_y + 2 * n_y * n_z + 2 * n_z * n_x
            if n_active_border_points / n_total_border_points > 0.5:
                continue

            alt_skeleton = skeletonize(the_particle_labels, method="lee")
            alt_distance = ndimage.distance_transform_edt(the_particle_labels)
            alt_skeleton[(alt_distance + 0.5) / n_y < MIN_CIRC_DIA / 2] = (
                0  # add 0.5 px due to discretization
            )
            alt_skel_dist = alt_skeleton * alt_distance
            alt_min_skel = alt_skeleton.copy()

            for indices in zip(*np.where(alt_skeleton)):
                this_skel_dist = alt_skel_dist[indices]
                if this_skel_dist <= 0:
                    continue
                dist_to_others = np.linalg.norm(index_grid - np.array(indices), axis=-1)
                overlapping = dist_to_others < this_skel_dist
                has_smaller_dist = alt_skel_dist < this_skel_dist
                # drop circles I overlap with that are smaller
                alt_skel_dist[overlapping & has_smaller_dist] = 0
                alt_min_skel[overlapping & has_smaller_dist] = 0
                # drop me if there are bigger distances overlapping with me: implicitly done in some other iteration
            alt_remaining_dists = alt_skel_dist[alt_skel_dist > 0]

            if is_2d:
                pass
                # fig, axes = plt.subplots(1, 4)
                # axes[0].imshow(the_particle_labels)
                # axes[1].imshow(alt_skeleton)
                # # axes[1].imshow(skel_dist, vmin=0, vmax=8)
                # # axes[1].imshow(recon_circles)
                # #axes[1].imshow(min_skel)
                # # axes[2].imshow(alt_skeleton)
                # # axes[2].imshow(alt_skel_dist, vmin=0, vmax=8)
                # axes[2].imshow(alt_min_skel)
                # # axes[3].imshow((mati[:, 0, :, :].transpose((1, 2, 0)) + 1) / 2)
                # # axes[3].hist(d_flat)
                # axes[3].set_xlabel(f"median {np.median(alt_remaining_dists)}")
                # axes[3].hist(alt_remaining_dists.flat)
                # # axes[4].set_xlabel(f"median {np.median(alt_remaining_dists)}")
                # plt.show()
            else:
                pass
                # fig, axes = plt.subplots(3, 3)
                # axes[0,0].imshow(the_particle_labels[4])
                # axes[0,1].imshow(the_particle_labels[8])
                # axes[0,2].imshow(the_particle_labels[12])
                # axes[1,0].imshow(the_particle_labels[16])
                # axes[1,1].imshow(the_particle_labels[20])
                # axes[1,2].imshow(the_particle_labels[24])
                # axes[2,0].imshow(alt_min_skel.sum(axis=0))
                # axes[2,2].hist(alt_remaining_dists.flat)
                # plt.show()

            remaining_dists = alt_remaining_dists

            if len(remaining_dists) == 0:
                continue
            if (
                (remaining_dists / n_y) > MAX_CIRC_DIA / 2
            ).any():  # should not be possible
                continue

            c_dia_dist = remaining_dists * 2 / n_y
            median = np.median(c_dia_dist)
            c_dia_medians[invert] = median
            c_dia_vars[invert] = c_dia_dist.var().item()

        if all([np.isinf(v) for v in c_dia_vars.values()]):  # matching circles failed
            # consider all as same material
            mat_matrix = mati_flat.mean(axis=0)  # (3,)
            mat_particles = mat_matrix
            mask_variance_sum = mati_flat.var(axis=0).sum()  # ()
            volume_fraction = 0
            circle_dia_frac = 0
            circle_dia_var = 0
            labels_return = np.zeros_like(labels)
        else:
            to_use = c_dia_vars[True] < c_dia_vars[False]
            circle_dia_frac = c_dia_medians[to_use]
            circle_dia_var = c_dia_vars[to_use]
            volume_fraction = {True: n_a / n_pxl, False: n_b / n_pxl}[to_use]
            the_particle_labels = (~labels) if to_use else labels
            matrix = mati_im[~the_particle_labels]
            particles = mati_im[the_particle_labels]
            mat_matrix = matrix.mean(axis=0)  # (3,)
            mat_particles = particles.mean(axis=0)  # (3,)
            assert mat_matrix.shape == (3,) and mat_particles.shape == (3,)
            mask_variance_sum = matrix.var(axis=0).sum() + particles.var(axis=0).sum()
            labels_return = the_particle_labels

    res = {
        "mat_matrix": mat_matrix,
        "mat_particles": mat_particles,
        "volume_fraction": volume_fraction,
        "circle_dia_frac": circle_dia_frac,
        "particle_labels": labels_return,
    }
    stats["mask_variance_sum"] = mask_variance_sum.item()
    stats["circle_dia_var"] = circle_dia_var

    stats["mat_nn_dist"] = matlist_data.query_distance(
        np.stack([mat_matrix, mat_particles])
    ).tolist()  # (2,) - list
    stats["matched_voxels"], matched_materials = matlist_data.match_to_voxels(
        np.stack([mat_matrix, mat_particles])
    )
    res["matched_mats"] = matched_materials
    res["matched_mat_matrix"], res["matched_mat_particles"] = matched_materials
    return res, stats


def check_mat_plausibility(mat: np.ndarray, ds):
    # silence convergence warnings on gmm.fit on bad data:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    matlist_data: MaterialListData = ds["matlist_data"]
    padding = ds["padding"]

    has_batch_dim = mat.ndim == 5  # (N, 3, n_x, n_y, n_z)
    if not has_batch_dim:
        mat = mat[None]
    mat = unpad(mat, padding)
    stat_results = defaultdict(list)
    material_stuffs = []

    for i in range(mat.shape[0]):
        mat_res, stat_res = fit_one(mat[i], matlist_data, fitting_seed=0)
        for k, v in stat_res.items():
            stat_results[k].append(v)
        material_stuffs.append(mat_res)

    return material_stuffs, stat_results


def gen_spheres(
    n_circles,
    sphere_radius,
    allow_intersections,
    rng: np.random.Generator,
    sampling_method,
    dim=3,
    max_tries=1e6,
):  # (N,dim)
    if sampling_method == "backtrack":
        total_tries = 0
        failed_at_level = [0]
        positions = np.empty((n_circles, dim))
        level = 0
        max_base_fails = 1e6
        max_level_fails = 4
        while level < n_circles:
            pos = rng.uniform(sphere_radius, 1 - sphere_radius, size=(dim,))
            total_tries += 1
            if level > 0:
                distances = np.linalg.norm(positions[:level] - pos, 2, axis=1)
                if (distances < 2 * sphere_radius).any():  # intersection -> fail!
                    if total_tries % 1e6 == 0:
                        print(
                            "for n_spheres=",
                            n_circles,
                            "total tries at ",
                            total_tries // 1e6,
                            "M. Currently at level",
                            level,
                            "restarting sampling.",
                        )
                        level = 0
                        failed_at_level = [failed_at_level[0] + 1]
                        continue
                    while level >= 0:
                        failed_at_level[level] += 1
                        if level == 0 and failed_at_level[0] % 1e4 == 0:
                            print("\tcurrent base fails", failed_at_level[0])
                        if failed_at_level[level] < (
                            max_level_fails if level > 0 else max_base_fails
                        ):
                            break
                        else:
                            failed_at_level.pop()
                            level -= 1
                    else:
                        raise RuntimeError(f"Too many base fails {max_base_fails}")
                    continue
            # all good
            positions[level] = pos
            level += 1
            failed_at_level.append(0)
        print("total tries for gen_spheres", total_tries)
        return np.array(positions)
    elif sampling_method == "forces":
        total_tries = 0
        while total_tries < max_tries:
            positions = rng.uniform(
                sphere_radius, 1 - sphere_radius, size=(n_circles, dim)
            )
            total_tries += 1
            push_steps = 0
            while push_steps < 1e4:
                # deltas[b,a]: vector from node a to node b
                deltas = positions[:, None, :] - positions[None, :, :]
                # print('pos', positions, 'deltas', deltas)
                base_distances = np.linalg.norm(deltas, 2, axis=-1, keepdims=True)
                outer_distances = base_distances - 2 * sphere_radius
                np.fill_diagonal(outer_distances[:, :, 0], 0)
                if (outer_distances >= 0).all():
                    break  # could also immediately return positions
                # there are penetrations
                directions = deltas / (base_distances + 1e-8)
                outer_distances[outer_distances < 0] = np.minimum(
                    -1e-3, outer_distances[outer_distances < 0]
                )  # ensure a certain moving distance
                correction_lengths = np.minimum(
                    0, outer_distances
                )  # only keeps negative, intersecting outer distances # TODO: more soft clipping to really push away from each other?
                noisy_directions = (
                    directions + rng.random((n_circles, n_circles, dim)) * 1e-6
                )  # incorporate tiny random part in case shperes collapse
                correction_deltas = (
                    noisy_directions * correction_lengths / 2
                )  # applying those pairwise would resolve pairwise collisions
                STEP_SIZE = 1.5
                actual_deltas = correction_deltas.sum(axis=0) * STEP_SIZE
                positions += actual_deltas
                positions = np.clip(positions, sphere_radius, 1 - sphere_radius)
                push_steps += 1

            else:  # could not resolve penetrations
                print("giving up after", push_steps, "push steps, now at", total_tries, "total tries for", n_circles, "circles of radius", sphere_radius)
                continue
            break
        else:
            raise RuntimeError(f"Too many fails {total_tries}")
        return positions


def float_pair(x):
    """'a,b' -> (a,b) and  'a' -> (a,a)"""
    res = tuple([float(q) for q in x.strip().split(",")])
    if len(res) == 2:
        return res
    elif len(res) == 1:
        return (res[0], res[0])
    else:
        raise ValueError("not an int or pair:" + str(res))


def gen_particle_structure(
    particle_vol_frac,
    circle_radius,
    allow_intersections,
    n_el_per_dim,
    element_coordinates_normalized,
    rng: np.random.Generator,
    dim,
    sample_instead_round=False,
    max_tries=1e6,
):
    area_mesh = 1
    if circle_radius == 0:
        assert particle_vol_frac == 0
        n_circles = 0
    else:
        if dim == 2:
            circle_vol = math.pi * circle_radius**2
        elif dim == 3:
            circle_vol = 4 / 3 * math.pi * circle_radius**3
        n_circles = particle_vol_frac * area_mesh / circle_vol
        if sample_instead_round:
            floor_, ceil_ = math.floor(n_circles), math.ceil(n_circles)
            p_ceil = n_circles - floor_
            n_circles = rng.choice(
                [floor_, ceil_],
                p=[1 - p_ceil.astype(np.float64), p_ceil.astype(np.float64)],
            )
        else:
            n_circles = round(n_circles)
    circle_positions = gen_spheres(
        n_circles,
        circle_radius,
        allow_intersections,
        rng,
        sampling_method="forces",
        dim=dim,
        max_tries=max_tries,
    )
    if dim == 2:
        is_particle = np.zeros(n_el_per_dim[1:], dtype=bool)
    elif dim == 3:
        is_particle = np.zeros(n_el_per_dim, dtype=bool)
    assert not allow_intersections, f"Periodic bound not implemented!"
    for pos in circle_positions:
        refpos = {2: pos[:, None, None], 3: pos[:, None, None, None]}[dim]
        dists = np.linalg.norm(element_coordinates_normalized - refpos, 2, axis=0)
        is_particle[dists <= circle_radius] = True
    return is_particle


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", "-nex", type=int, default=100_000)
    parser.add_argument(
        "--material_sampling",
        type=str,
        choices=["random", "list_chunks", "list_nochunks"],
    )
    parser.add_argument(
        "--material_normalization_path",
        type=str,
        default="material_normalization.pkl",
        help="only for list materials. where to take the (normalization) data from.",
    )
    parser.add_argument(
        "--e_uniform",
        type=float_pair,
        default=(50e9, 150e9),
        help="uniform e range. log see next.",
    )
    parser.add_argument(
        "--e_log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to sample E uniformly in log or linear space.",
    )
    parser.add_argument(
        "--poisson_uniform",
        type=float_pair,
        default=(0.25, 0.35),
        help="uniform poisson range",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="output dir name for split format / output .npz file name for all in one",
    )
    parser.add_argument(
        "--volume_fraction_uniform",
        type=float_pair,
        default=(0.3, 0.3),
        help="uniform fraction of material that is particle",
    )
    parser.add_argument(
        "--circle_diameter_uniform",
        type=float_pair,
        default=(0.3, 0.3),
        help="diameter determined as fraction of whole size",
    )
    parser.add_argument("--num_edge_nodes", type=int)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="number of workers stared as parallel threads",
    )
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument(
        "--base_config", help=".ini file to base this on. contradicts num_edge_nodes"
    )
    parser.add_argument("--fem_path", type=str, default="FEM3D/fem3D")
    parser.add_argument("--no_tqdm", action="store_true")
    parser.add_argument("--override_mat_seed", type=int)
    parser.add_argument("--dim", type=int, default=2)

    args = parser.parse_args()
    with_base = args.base_config is not None
    if with_base:
        assert (
            args.num_edge_nodes is None
        ), f"Cannot give base config and specify topology!"
        base_config = read_in_file_3d(args.base_config)
        # determine structure properties in base config
        # assume regular grid mesh
        (
            n_nodes_per_dim,
            min_coord_per_dim,
            max_coord_per_dim,
            coord_delta_per_dim,
            coords_per_dim,
        ) = ({}, {}, {}, {}, {})
        for d in range(3):
            d_values = np.unique(base_config.coordinates[:, d])
            # assert regular grid
            deltas = d_values[1:] - d_values[:-1]
            assert np.isclose(
                deltas, deltas[0], atol=1e-4
            ).all(), f"deltas err {deltas}"
            n_nodes_per_dim[d], coords_per_dim = len(d_values), d_values
            min_coord_per_dim[d], max_coord_per_dim[d], coord_delta_per_dim[d] = (
                d_values[0],
                d_values[-1],
                deltas[0],
            )
        _n_nodes = np.prod(list(n_nodes_per_dim.values()))
        assert (
            _n_nodes == base_config.n_nodes
        ), f"Config says {base_config.n_nodes}, but prod of per dim yields {_n_nodes}"
        n_el_per_dim = {d: n_nodes_per_dim[d] - 1 for d in range(3)}
        assert np.prod(list(n_el_per_dim.values())) == base_config.n_elements

        # assert node order
        coords_structured = structure_nodes(
            base_config.coordinates.T, n_nodes_per_dim
        )  # (d=3, n_nodes) --> (d=3, w, h, depth)
        assert (coords_structured[:, :-1, :, :] <= coords_structured[:, 1:, :, :]).all()
        assert (coords_structured[:, :, :-1, :] <= coords_structured[:, :, 1:, :]).all()
        assert (coords_structured[:, :, :, :-1] <= coords_structured[:, :, :, 1:]).all()
        assert (flatten_nodes(coords_structured) == base_config.coordinates.T).all()

        # element coordinates
        element_coordinates = base_config.coordinates[base_config.elements, :]
        assert element_coordinates.ndim == 3
        element_coordinates = element_coordinates.mean(
            axis=1
        )  # mean over nodes per el --> (n_el, 3)

        # assert element order (just use node functions)
        el_coords_struct = structure_nodes(element_coordinates.T, n_el_per_dim)
        assert (el_coords_struct[:, :-1, :, :] <= el_coords_struct[:, 1:, :, :]).all()
        assert (el_coords_struct[:, :, :-1, :] <= el_coords_struct[:, :, 1:, :]).all()
        assert (el_coords_struct[:, :, :, :-1] <= el_coords_struct[:, :, :, 1:]).all()
        elements = base_config.elements
        coordinates = base_config.coordinates
        nnpd = tuple(n_nodes_per_dim[k] for k in [0, 1, 2])
        n_el_per_dim = tuple(n_el_per_dim[k] for k in [0, 1, 2])
        if args.dim == 2:
            elem_coords = el_coords_struct[1:, 0, :, :]  # the 2D part
        elif args.dim == 3:
            elem_coords = el_coords_struct[:, :, :, :]  # everything
        mesh_scale = coords_structured[1, 0, -1, 0]
        assert coords_structured[2, 0, 0, -1] == mesh_scale  # square

    else:  # build mesh
        assert args.num_edge_nodes is not None
        n_edge = args.num_edge_nodes
        n_el_edge = n_edge - 1

        if args.dim == 3:
            raise NotImplementedError

        # node coords
        edge = np.linspace(0, 1, num=n_edge, endpoint=True)  # everything in [0,1]
        delta_node = 1 / (n_edge - 1)
        node_coords = np.stack(np.meshgrid(edge, edge, indexing="ij"), axis=0)  # 2,N,N
        elem_coords = node_coords[:, :-1, :-1] + delta_node / 2

        # 3D variants
        coordinates_struct = np.empty((3, 2, n_edge, n_edge))  # (3,2,N,N)
        coordinates_struct[:2] = node_coords
        coordinates_struct[2, 0] = 0  # introduce x-axis
        coordinates_struct[2, 1] = 1 / n_el_edge  # make cubes

        # build mesh
        n_el_per_dim = (1, n_el_edge, n_el_edge)
        n_el = np.prod(n_el_per_dim)
        nnpd = (2, n_edge, n_edge)  # num nodes per dim
        n_nodes = np.prod(nnpd)
        elements = np.ones((n_el, 8), dtype=np.int32) * (-1)
        indices = np.arange(n_nodes)  # (n_nodes,)
        indices_struct = structure_nodes(indices[None], nnpd)[0]  # (2,N,N)
        i_el = 0
        for iz, iy, ix in itertools.product(
            range(n_el_per_dim[2]), range(n_el_per_dim[1]), range(n_el_per_dim[0])
        ):
            ib = indices_struct[ix, iy, iz]
            elements[i_el] = [
                ib,
                ib + 1,
                ib + nnpd[0] + 1,
                ib + nnpd[0],
                ib + nnpd[0] * nnpd[1],
                ib + nnpd[0] * nnpd[1] + 1,
                ib + nnpd[0] * nnpd[1] + nnpd[0] + 1,
                ib + nnpd[0] * nnpd[1] + nnpd[0],
            ]
            i_el += 1
        assert i_el == n_el
        coordinates = flatten_nodes(coordinates_struct).T  # (3,2,N,N) --> (n_nodes,3)
        mesh_scale = 1

    def gen_example(seed):
        rng = np.random.default_rng(seed)
        if args.material_sampling == "random":
            if args.e_log:
                e_min_log, e_max_log = (
                    np.log(args.e_uniform[0]),
                    np.log(args.e_uniform[1]),
                )
                e_matrix_log, e_particle_log = rng.uniform(
                    e_min_log, e_max_log, size=(2,)
                )
                e_matrix, e_particle = np.e**e_matrix_log, np.e**e_particle_log
            else:
                e_matrix, e_particle = rng.uniform(*args.e_uniform, size=(2,))
            nu_matrix, nu_particle = rng.uniform(*args.poisson_uniform, size=(2,))
            rhos = [np.ones_like(nu_matrix) * -1, np.ones_like(nu_matrix) * -1]
            materials = np.array(
                [[e_matrix, nu_matrix, rhos[0]], [e_particle, nu_particle, rhos[1]]]
            )
        elif args.material_sampling.startswith("list"):
            with open(args.material_normalization_path, "rb") as f:
                matlist_data: MaterialListData = pickle.load(f)
            if args.material_sampling == "list_nochunks":
                sample_fn = matlist_data.sample_material_nochunks
            else:
                sample_fn = matlist_data.sample_material_chunks
            _, mat_matrix_unnorm = sample_fn(rng)
            _, mat_particle_unnorm = sample_fn(rng)
            if args.override_mat_seed is not None:
                override_rng = np.random.default_rng(args.override_mat_seed)
                _, mat_matrix_unnorm = sample_fn(override_rng)
                _, mat_particle_unnorm = sample_fn(override_rng)
            materials = np.stack([mat_matrix_unnorm, mat_particle_unnorm], axis=0)

        # keep these fixed per sample
        particle_vol_frac = rng.uniform(*args.volume_fraction_uniform)
        circle_radius = rng.uniform(*args.circle_diameter_uniform) / 2  # dia -> radius

        allow_intersections = False
        element_coordinates_normalized = elem_coords / mesh_scale

        is_particle = gen_particle_structure(
            particle_vol_frac,
            circle_radius,
            allow_intersections,
            n_el_per_dim,
            element_coordinates_normalized,
            rng,
            dim=args.dim,
        )
        actual_vol_frac = is_particle.sum() / is_particle.size
        mat_ass = is_particle.astype(int)

        # make 3D
        if args.dim == 2:
            mat_ass = mat_ass[None, :, :]  # (1, N-1, N-1)
        mat_ass = flatten_nodes(mat_ass[None])[0]  # (n_el,)

        params = {
            "materials": materials,
            "mat_ass": mat_ass,
            "seed": np.array(seed),
            "particle_vol_frac": particle_vol_frac,
            "circle_radius": circle_radius,
            "actual_vol_frac": actual_vol_frac,
        }

        if with_base:  # this also means running the solver!
            config = deepcopy(base_config)
            config.n_materials = 2
            if materials[0, 2] == -1:  # not defined, e.g. due to random
                # -> copy from config
                assert (
                    config.material_data[:, 2] == config.material_data[0, 2]
                ).all(), f"Different density for materials, I don't know which one to choose!"
                materials[:, 2] = config.material_data[0, 2]
            config.material_data = materials
            config.element_material = mat_ass

            f = NamedTemporaryFile("w", suffix=f"_{seed}.ini", delete=False)
            try:
                write_in_file_3d(config, f)  # closes
                outfile = f.name.replace(".ini", ".npz")
                res = subprocess.run(
                    [
                        args.fem_path,
                        f.name,
                        "--solver=direct",
                        "--export_npz",
                    ]
                )
                if res.returncode != 0:
                    print("subproc failed! stderr:", res.stderr)
                    raise RuntimeError()
                with np.load(outfile) as outd_np:
                    outd = {k: v.copy() for k, v in outd_np.items()}
                os.remove(outfile)
                os.remove(f.name.replace(".ini", ".out"))
            finally:
                # for debugging, keep config files:
                if seed in []:  # put numbers here
                    print("--- Not deleting ini file:", f.name)
                else:
                    os.remove(f.name)
        else:
            outd = {}

        return params, outd

    all_things = defaultdict(lambda: [None for _ in range(args.num_examples)])

    if args.name.endswith(".npz"):
        all_in_one = True
    elif args.name.endswith(".npy"):
        raise ValueError()
    else:
        all_in_one = False
        os.mkdir(args.name)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.num_workers
    ) as executor:
        futures = [
            executor.submit(gen_example, seed=i + args.base_seed)
            for i in range(args.num_examples)
        ]
        tqdm_minint, tqdm_maxint = (0.1, 10) if sys.stderr.isatty() else (60, 300)
        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures),
            desc="Generating examples",
            total=args.num_examples,
            disable=args.no_tqdm,
            mininterval=tqdm_minint,
            maxinterval=tqdm_maxint,
        ):  # timeout=600
            params, out = future.result()
            i = params["seed"] - args.base_seed
            if (
                not "const_btype" in all_things and "btype" in out
            ):  # we got a FEM running and are the first one
                all_things["const_btype"] = out["btype"]
                all_things["const_known_displacements"] = out["known_displacements"]

            for k, v in params.items():
                all_things[k][i] = v
            for k in list(params):
                del params[k]

            for k in ["x", "AveK"]:
                all_things[k][i] = out[k]

            for k in list(out):
                del out[k]
            del out

    all_things = {k: np.stack(v) for k, v in all_things.items()}
    all_things["const_elements"] = elements
    all_things["const_coordinates"] = coordinates
    all_things["const_n_nodes_per_dim"] = nnpd
    all_things["const_ds_hint"] = "particles"

    print(
        "materials:",
        all_things["materials"].shape,
        "mat_ass:",
        all_things["mat_ass"].shape,
        "const_elements:",
        all_things["const_elements"].shape,
    )
    if "x" in all_things:
        print("x:", all_things["x"].shape)
    print(
        f"Average K: mean {all_things['AveK'].mean():>6.2f} GPa  stddev {all_things['AveK'].std():>6.2f} GPa"
    )

    for k, v in all_things.items():
        if isinstance(v, np.ndarray):
            assert not np.isnan(v).any() and not np.isinf(v).any()
    if all_in_one:
        np.savez_compressed(args.name, **all_things)
    else:
        np.savez_compressed(args.name + "/others.npz", **all_things)


if __name__ == "__main__":
    run()
