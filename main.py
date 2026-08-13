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
from collections import defaultdict
import gc
import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch as th
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import wandb
from scipy.stats import gaussian_kde

from dataset_utils import flatten_nodes, get_dataset, get_x_and_cond
from spherical_composite_dataset import gen_particle_structure, check_mat_plausibility
from diffusion.unet3d import MatUNet3DModel
from diffusion.pipelines import NDConditionalDDPMPipeline
from diffusion.ddpm import GuidanceDDPMScheduler
from material_normalization import MaterialListData
import solver_interface
from solver_interface import SolverInterface


from global_config import config as glconf, NotADict

from inspect_dataset import plot_img


def evaluate_target(samples: np.ndarray, aligned_solvers: bool):
    bs = samples.shape[0]
    interfaces = solver_interface.INTERFACES
    x_pre, _ = interfaces[0].prepare_send(samples)  # -> (bs,d,N)

    grads_raw, Ks, Js, it_counts = solver_interface.communicate(
        x_pre, aligned_solvers, calc_grad=False, precise=False
    )  # precise?
    densities_unnorm = x_pre[:, 2]
    avg_density = densities_unnorm.mean(axis=1)
    assert avg_density.shape == (bs,)
    return Ks, Js, avg_density


def evaluate_material_target(
    material_stuff_list,
    ds_stuff,
    aligned_solvers,
    also_closest_existing=True,
    num_spatial_samples=5,
):
    interfaces = solver_interface.INTERFACES
    element_coordinates_normalized = interfaces[0].element_coordinates_normalized
    n_el_per_dim = interfaces[0].n_el_per_dim
    dim = 2 if n_el_per_dim[0] == 1 else 3
    matlist_data: MaterialListData = ds_stuff["matlist_data"]
    bs = len(material_stuff_list)
    n = np.prod(n_el_per_dim)

    def calc(closest=False):
        Ks_all, Js_all, dens_all = [], [], []

        for i_spatial in range(num_spatial_samples):
            mats_flat = []

            for i_thing, mat_stuff in enumerate(material_stuff_list):
                rng = np.random.default_rng(i_spatial)
                try:
                    is_particle = gen_particle_structure(
                        mat_stuff["volume_fraction"],
                        mat_stuff["circle_dia_frac"] / 2,
                        False,
                        n_el_per_dim,
                        element_coordinates_normalized,
                        rng,
                        dim=dim,
                        sample_instead_round=True,
                        max_tries=5,
                    )
                except RuntimeError:  # tries exceeded
                    print(
                        "Warning! Could not generate particle structure for vf",
                        mat_stuff["volume_fraction"],
                        "dia frac",
                        mat_stuff["circle_dia_frac"],
                        "-- assuming no particles now.",
                    )
                    if dim == 2:
                        is_particle = np.zeros(n_el_per_dim[1:], dtype=bool)
                    elif dim == 3:
                        is_particle = np.zeros(n_el_per_dim, dtype=bool)
                if dim == 2:
                    is_particle = is_particle[None]  # -> (1, n_y, n_z)

                mat_matrix, mat_particles = (
                    mat_stuff["mat_matrix"],
                    mat_stuff["mat_particles"],
                )
                if closest:
                    mat_matrix, mat_particles = mat_stuff["matched_mats"]

                this_mat = np.where(
                    is_particle[:, :, :, None], mat_particles, mat_matrix
                )

                this_mat = this_mat.transpose((-1, 0, 1, 2))  # in front
                this_mat_flat = flatten_nodes(this_mat)
                mats_flat.append(this_mat_flat)

            mats_flat = np.stack(mats_flat)
            assert mats_flat.shape == (bs, 3, n), f"{mats_flat.shape=}"
            mats_flat = mats_flat.transpose((0, 2, 1))  # (bs, n, d)
            mats_flat_unn = matlist_data.unnormalize(mats_flat)
            mats_flat_unn = mats_flat_unn.transpose((0, 2, 1))  # (bs, d, n)

            grads_raw, Ks, Js, it_counts = solver_interface.communicate(
                mats_flat_unn,
                aligned_solvers,
                calc_grad=False,
                keep_sol=False,
                precise=True,
            )

            Ks_all.append(Ks)
            Js_all.append(Js)

            densities_unnorm = mats_flat_unn[:, 2]
            avg_density = densities_unnorm.mean(axis=1)
            assert avg_density.shape == (bs,)
            dens_all.append(avg_density)

        Ks_all, Js_all, dens_all = (
            np.stack(Ks_all),
            np.stack(Js_all),
            np.stack(dens_all),
        )
        K_avg, J_avg, dens_avg = (
            Ks_all.mean(axis=0),
            Js_all.mean(axis=0),
            dens_all.mean(axis=0),
        )
        K_var = Ks_all.var(axis=0)
        s_closest = "closest_" if closest else ""
        res = {
            f"K_{s_closest}spatial_avg": K_avg,
            f"J_{s_closest}spatial_avg": J_avg,
            f"density_{s_closest}spatial_avg": dens_avg,
            f"K_{s_closest}spatial_var": K_var,
        }
        assert K_avg.shape == (bs,) and K_var.shape == (bs,)
        return res

    main_res = calc()
    if also_closest_existing:
        ress_ = calc(closest=True)
        main_res.update(ress_)

    return main_res


def evaluate(
    config,
    val_ds,
    ds_stuff,
    pipeline: NDConditionalDDPMPipeline,
    conditional,
    global_step=None,
    seed=0,
):
    pipeline.unet.eval()
    plotfig = None
    _time = time.time()

    if config["save_samples"]:
        all_samples = []
        all_matstuff = []
    gp = config["guidance_params"]

    all_intermediates = []

    with torch.inference_mode():
        res_per = defaultdict(list)
        count = config["num_val_batches"]
        it_count_per_total = np.array(0.0)
        timestats = defaultdict(float)
        eval_rng = torch.Generator(device="cpu").manual_seed(seed)
        bs = config["val_batch_size"]
        solver_calls_per = None
        for ib in range(count):
            c_batch, ds_indices = None, None
            if conditional is not None:
                if config["inverse_target"] is not None:
                    c_batch = (
                        torch.ones(
                            (bs, 1), device=pipeline.unet.device, dtype=torch.float32
                        )
                        * config["inverse_target"]
                        / 700
                    )  # normalized
                else:
                    stuff = []
                    # sample stuff from fitted training set
                    for cond_kde in conditional:
                        cond_kde: gaussian_kde = cond_kde
                        stuff.append(
                            cond_kde.resample(config["val_batch_size"], seed=ib).T
                        )  # sample similarly to training
                    stuff = np.concatenate(stuff, axis=1)
                    c_batch = torch.from_numpy(stuff).to(
                        pipeline.unet.device, torch.float32
                    )

            # debug stuff
            return_n_intermediate = 0
            return_n_intermediate = int(os.environ.get("N_INTERMEDIATE", 0))
            pred_orig_as_intermediate = True

            num_inference_steps = config["timesteps"]
            _time_pipeline = time.time()
            samples, intermediate_samples, intermediate_timesteps, sampling_stats = (
                pipeline(
                    cond=c_batch,
                    batch_size=config["val_batch_size"],
                    generator=eval_rng,
                    num_inference_steps=num_inference_steps,
                    return_dict=False,
                    ds_indices=ds_indices,
                    return_n_intermediate=return_n_intermediate,
                    pred_orig_as_intermediate=pred_orig_as_intermediate,
                )
            )
            samples.cpu().numpy()  # enforce cpu
            timestats["t_pipeline_sum"] += time.time() - _time_pipeline

            # print('solver its per item', sampling_stats['it_count_sum'] / bs)
            it_count_per_total += sampling_stats["it_count_sum"] / bs
            solver_calls_per = sampling_stats["solver_calls_per_sum"]
            # NOT anymore per batch size!
            for k, v in sampling_stats.items():
                if k.startswith("t_"):
                    timestats[k] += v  # / bs

            if len(intermediate_samples) > 0:
                intermediate_samples = torch.stack(intermediate_samples)
                intermediate_samples = np.clip(
                    intermediate_samples.cpu().numpy(), -1, 1
                )
                all_intermediates.append(intermediate_samples)

            if config["learn"] == "mat":
                samples_np_orig = samples.cpu().numpy()

                bs, d, *stuff = samples_np_orig.shape
                n_pxl = np.prod(stuff)
                b_oor = ((samples_np_orig > 1) | (samples_np_orig < -1)).any(axis=1)
                n_oor_per = b_oor.reshape(bs, -1).sum(axis=1)
                oor_frac_per = n_oor_per / n_pxl
                if (oor_frac_per > 0.01).any():
                    print(
                        "warning: high oor fracs in",
                        (oor_frac_per > 0.01).sum(),
                        "of",
                        bs,
                        "samples",
                    )
                res_per["oor_frac"].extend(oor_frac_per.tolist())

                samples_np = np.clip(samples_np_orig, -1, 1)
                print("checking mat..")
                material_stuff, result = check_mat_plausibility(samples_np, ds_stuff)
                for k, v in result.items():
                    res_per[k].extend(v)

                if config["save_samples"]:
                    all_samples.append(samples_np)
                    all_matstuff.extend(material_stuff)

                ### Inverse target stuff ###
                K = None
                if config["inverse_target"] is not None and not config["no_eval"]:
                    ## TODO: reset iterative solvers here!!
                    K, J, avg_density = evaluate_target(
                        samples_np, aligned_solvers=config["aligned_solvers"]
                    )
                    res_per["K_direct"].extend(K.tolist())
                    res_per["J_direct"].extend(J.tolist())
                    res_per["density_direct"].extend(avg_density.tolist())

                    eval_res_target = evaluate_material_target(
                        material_stuff,
                        ds_stuff,
                        aligned_solvers=config["aligned_solvers"],
                        also_closest_existing=True,
                        num_spatial_samples=config["num_spatial_samples"],
                    )
                    for k, v in eval_res_target.items():
                        res_per[k].extend(v)
            else:
                raise ValueError(f"Cannot learn {config['learn']}")

            if plotfig is None:
                dimensionality = 3 - len(config["ignore_dims"])
                N_PLOT = 12
                if config["learn"] == "mat":
                    N_PLOT = min(10, bs)
                    pred_mat = samples_np  # without oor markers
                    captions = None
                    titles = []

                    if "K" in sampling_stats and K is not None:
                        captions = [
                            [
                                f"{k:.1f} / {k_a:.1f} / {k_cl:.1f}"
                                for k, k_a, k_cl in zip(
                                    K,
                                    eval_res_target["K_spatial_avg"],
                                    eval_res_target["K_closest_spatial_avg"],
                                )
                            ]
                        ]  # two parentheses because of intermediate samples
                    else:
                        captions = [["none"] * N_PLOT]

                    if len(intermediate_samples) > 0:
                        pred_mat_inter = [
                            sample[:N_PLOT] for sample in intermediate_samples
                        ]
                        pred_mat = pred_mat_inter + [pred_mat]
                        titles = [str(q) for q in intermediate_timesteps] + ["final"]
                        captions = [
                            ["" for r in range(N_PLOT)]
                            for s in range(len(intermediate_samples))
                        ] + captions

                    plotfig = plot_img(
                        N_PLOT,
                        pred_mat=pred_mat,
                        titles=titles,
                        dimensionality=dimensionality,
                        captions=captions,
                    )  # min,max in sigmas
                else:
                    raise ValueError()

        res_acc = {}

        def aggregate(result_thing, acc_thing):
            if "K_direct" in result_thing and "K_closest_spatial_avg" in result_thing:
                K_abs_diff = np.abs(
                    np.array(result_thing["K_direct"])
                    - np.array(result_thing["K_closest_spatial_avg"])
                )
                K_rel_diff = K_abs_diff / config["inverse_target"]
                result_thing["K_reldiff"] = K_rel_diff.tolist()
                acc_thing["K_reldiff_mean"] = K_rel_diff.mean().item()
                acc_thing["K_reldiff_median"] = np.median(K_rel_diff).item()
            for k, v in result_thing.items():
                if k in ["K_reldiff"]:
                    continue
                if k in ["K_closest_spatial_avg"]:
                    arr = np.array(v)
                    K_rel_err = (
                        np.abs(arr - config["inverse_target"])
                        / config["inverse_target"]
                    )
                    acc_thing["K_relerr_mean"] = K_rel_err.mean().item()
                    acc_thing["K_relerr_median"] = np.median(K_rel_err).item()
                    for margin, margin_name in zip(
                        [0.01, 0.02, 0.05, 0.1, 0.5], ["1%", "2%", "5%", "10%", "50%"]
                    ):
                        acc_thing[f"K_relerr_{margin_name}"] = (
                            (K_rel_err < margin).sum() / arr.size
                        ).item()
                if k in [
                    "fit_variance_sum",
                    "mask_variance_sum",
                    "recon_mask_variance_sum",
                    "mat_nn_dist",
                    "circle_dia_median",
                    "circle_dia_var",
                    "K_direct",
                    "J_direct",
                    "density_direct",
                    "K_spatial_avg",
                    "J_spatial_avg",
                    "density_spatial_avg",
                    "K_spatial_var",
                    "K_closest_spatial_avg",
                    "J_closest_spatial_avg",
                    "density_closest_spatial_avg",
                    "K_closest_spatial_var",
                    "oor_frac",
                    "matrix_dist_",
                    "matrix_dist_closest_",
                    "matrix_dist_norm_",
                    "matrix_dist_norm_closest_",
                ]:
                    acc_thing[k] = np.array(v).mean().item()
                elif k in ["matched_voxels"]:
                    v = np.array(v)
                    assert v.ndim == 3 and v.shape[1] == 2 and v.shape[2] == 3
                    result_thing[k] = v.tolist()
                    v = v.reshape(-1, 3)
                    unique_voxels = np.unique(v, axis=0)
                    print("matched", len(unique_voxels), "unique voxels")
                    matlist_data: MaterialListData = ds_stuff["matlist_data"]
                    acc_thing["voxel_coverage"] = (
                        len(unique_voxels) / matlist_data.n_nonempty_voxels
                    )
                else:
                    raise ValueError("idk " + str(k))

        aggregate(res_per, res_acc)

        res_acc["it_count_avg"] = (
            it_count_per_total / config["num_val_batches"]
        ).item()
        res_acc["solver_calls_per"] = solver_calls_per
        for k, v in timestats.items():
            res_acc[k] = v / config["num_val_batches"]

        if config["save_samples"]:
            all_samples = np.concatenate(all_samples, axis=0)
            samples_paths = os.path.join(config["results_folder"], "samples.npz")
            np.savez_compressed(samples_paths, samples=all_samples)
            matstuff_red = defaultdict(list)
            matstuff_save_keys = [
                "mat_matrix",
                "mat_particles",
                "volume_fraction",
                "circle_dia_frac",
                "matched_mat_matrix",
                "matched_mat_particles",
            ]  # exclude: 'particle_labels', 'matched_mats'. TODO include?
            for matstuff in all_matstuff:
                for k, v in matstuff.items():
                    if k in matstuff_save_keys:
                        if isinstance(v, np.ndarray):
                            v = v.item() if v.size == 1 else v.tolist()
                        elif isinstance(v, (np.float32, np.float64)):
                            v = float(v)
                        matstuff_red[k].append(v)
            matstuff_path = os.path.join(config["results_folder"], "matstuff.json")
            with open(matstuff_path, "w") as f:
                json.dump(matstuff_red, f)

    _time = time.time() - _time
    res_acc["eval_time"] = _time

    if len(all_intermediates) > 0:
        samples_paths = os.path.join(config["results_folder"], "intermediates.npz")
        all_intermediates = np.concatenate(all_intermediates, axis=1)
        np.savez_compressed(
            samples_paths,
            intermediates=all_intermediates,
            timesteps=np.array(intermediate_timesteps),
        )
        print("saved intermediates to", samples_paths)

    pipeline.unet.train()
    pytorch_gc()  # helps a bit with GPU memory

    return res_acc, res_per, None, plotfig


def cycle(dl):
    while True:
        for data in dl:
            yield data


def train_loop(config, train_ds, val_ds, ds_thing, model, sched, pipeline, conditional):
    from diffusers.optimization import (
        get_cosine_schedule_with_warmup,
        get_constant_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    if config["lr_sched"] == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config["lr_warmup_steps"],
            num_training_steps=config["train_steps"],
        )
    elif config["lr_sched"] == "constant":
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=config["lr_warmup_steps"]
        )
    elif config["lr_sched"] == "polynomial":
        lr_scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=config["lr_warmup_steps"],
            num_training_steps=config["train_steps"],
            lr_end=config["lr"] / 1e2,
            power=0.5,  # TODO check?
        )
    else:
        raise ValueError("unknown" + str(config["lr_sched"]))

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    cycle_train_dl = cycle(train_dl)
    target_steps = config["train_steps"]

    global_step = 0

    tqdm_minint, tqdm_maxint = (0.1, 10) if sys.stderr.isatty() else (60, 300)
    progress_bar = tqdm(
        total=target_steps,
        disable=False,
        mininterval=tqdm_minint,
        maxinterval=tqdm_maxint,
    )
    progress_bar.set_description(f"Training")
    noise_scheduler: GuidanceDDPMScheduler = sched
    drop_cond_prob = config["drop_cond_prob"]

    # from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
    # derive loss weight
    # snr - signal noise ratio

    snr = noise_scheduler.alphas_cumprod / (1 - noise_scheduler.alphas_cumprod)
    # https://arxiv.org/abs/2303.09556
    maybe_clipped_snr = snr.clone()
    if False:  # min_snr_loss_weight
        maybe_clipped_snr.clamp_(max=min_snr_gamma)
    loss_weight = {
        "epsilon": maybe_clipped_snr / snr,
        "sample": maybe_clipped_snr,
        "v": maybe_clipped_snr / (snr + 1),
    }[config["prediction"]]
    if not config["snr_loss_weighting"]:
        loss_weight = th.ones_like(loss_weight)
    loss_weight = loss_weight.to(model.device)

    def mse_batch(y_hat, y_target, weighting=None):
        bs = y_hat.shape[0]
        y_hat, y_target = y_hat.reshape(bs, -1), y_target.reshape(bs, -1)
        loss = ((y_hat - y_target) ** 2).mean(dim=1)
        if weighting is not None:
            loss *= weighting
        loss = loss.mean(dim=0)
        return loss

    model.train()
    res_acc = None

    while global_step < target_steps:
        if conditional:
            x, c = next(cycle_train_dl)
        else:
            (x,) = next(cycle_train_dl)
            c = None

        # Sample noise to add to the images
        noise = torch.randn(x.shape, device=x.device)
        bs = x.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=x.device,
            dtype=torch.int64,
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(x, noise, timesteps)

        (pred,) = model(
            noisy_images, c, timesteps, return_dict=False, drop_cond_prob=drop_cond_prob
        )
        this_loss_weight = loss_weight[timesteps]
        if config["prediction"] == "epsilon":
            loss = mse_batch(pred, noise, this_loss_weight)
        elif config["prediction"] == "sample":
            loss = mse_batch(pred, x, this_loss_weight)
        elif config["prediction"] == "v":
            v_samples = noise_scheduler.get_velocity(x, noise, timesteps)
            loss = mse_batch(pred, v_samples, this_loss_weight)
        else:
            raise NotImplementedError()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        pipeline.ema.update()

        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

        global_step += 1

        if global_step != 0 and (global_step % config["eval_every"] == 0):
            print("saving model before evaluation at step", global_step)
            save_dict = {
                "model": model.state_dict(),
                "ema": pipeline.ema.state_dict(),
            }
            th.save(save_dict, os.path.join(config["results_folder"], "trained.pt"))
            res_acc, res_per, res_acc_ds, plotfig = evaluate(
                config,
                val_ds,
                ds_thing,
                pipeline,
                conditional=conditional,
                global_step=global_step,
                seed=config["eval_seed"],
            )
            with open(
                os.path.join(
                    config["results_folder"], f"eval_res_acc_{global_step}.json"
                ),
                "w",
            ) as f:
                json.dump(res_acc, f)
            with open(
                os.path.join(config["results_folder"], f"eval_res_per_last.json"), "w"
            ) as f:
                json.dump(res_per, f)
            logs.update(res_acc)
            if plotfig is not None:
                plotfig_path = os.path.join(
                    config["results_folder"], f"plot_{global_step}.png"
                )
                plotfig.savefig(plotfig_path)
                logs["fig"] = wandb.Image(plotfig_path)
            print(f"step {global_step}, eval res:", res_acc)
            model.train()  # just to be sure
        if config["wandb"]:
            wandb.log(logs, step=global_step)
        # else:
        #     print(logs)

        # pipeline.save_pretrained(config.results_folder)
    return res_acc


def inttuple(x):
    return tuple([int(q) for q in x.strip().split(",")])


def ensure_tensor(x):
    return x if th.is_tensor(x) else th.tensor(x)


class TinyNormalizer(th.nn.Module):
    def __init__(self, x_mean, x_std, c_mean=0, c_std=-1):
        th.nn.Module.__init__(self)
        self.register_buffer("x_mean", ensure_tensor(x_mean))
        self.register_buffer("x_std", ensure_tensor(x_std))
        self.register_buffer("c_mean", ensure_tensor(c_mean))
        self.register_buffer("c_std", ensure_tensor(c_std))

    def get(self, to=None, numpy=False):
        if numpy and to is None:
            to = ("cpu",)
        stuff = {k: getattr(self, k) for k in ["x_mean", "x_std", "c_mean", "c_std"]}
        if to is not None:
            stuff = {k: v.to(*to) for k, v in stuff.items()}
        if numpy:
            stuff = {k: v.numpy() for k, v in stuff.items()}
        return SimpleNamespace(**stuff)


def best_type(x):
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return x


def str2dict(s):
    return {(kv := q.split("="))[0]: best_type(kv[1]) for q in s.split(",")}


def pytorch_gc():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


def run(sweep_dict={}):
    # global test_SI

    parser = argparse.ArgumentParser()
    parser.add_argument("--load")
    parser.add_argument("--timesteps", default=1000, type=int, help="INFERENCE (generation) timesteps. Training always with 1000.")
    parser.add_argument("--train_steps", default=10_000, type=int)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--lr_sched", default="cosine", choices=["constant", "cosine", "polynomial"])
    parser.add_argument("--lr_warmup_steps", default=5000, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--val_batch_size", type=int, default=512)
    parser.add_argument("--num_val_batches", type=int, default=5)
    parser.add_argument("--ds", default="data.npz")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--eval_every", default=5000, type=int)
    parser.add_argument("--guidance_scale", default=1.0, type=float, help="For classifier-free guidance (sampling only)")
    parser.add_argument("--guidance_rescale", default=0.0, type=float, help="For classifier-free guidance (sampling only)")
    parser.add_argument("--drop_cond_prob", default=0.0, type=float, help="For classifier-free guidance (training only)")
    parser.add_argument("--conditional", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dir")
    parser.add_argument("--runs_dir", help="use to build directory from name")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="β_start")
    parser.add_argument("--beta_max", type=float, default=2e-3, help="β_end")
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--rescale_betas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timestep_spacing", type=str, choices=["linspace", "leading", "trailing"], default="trailing")
    parser.add_argument("--prediction", type=str, choices=["epsilon", "sample", "v"], default="v")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seed", type=int, default=0)
    parser.add_argument("--append_seed_name", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--downsample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ignore_dims", type=inttuple, default=())
    parser.add_argument("--hidden_sizes", type=inttuple, default=(16, 32), help="hidden dimension of outer blocks")
    parser.add_argument("--num_outer_layers", type=int, default=2, help="number of layers per block")
    parser.add_argument("--mid_hidden_sizes", type=inttuple, default=(64, 64), help="number of layers for mid block")
    parser.add_argument("--time_embedding", type=str, default="fourier")
    parser.add_argument("--cond_net", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--downsample_type", choices=["avgpool", "conv"], default="avgpool")
    parser.add_argument("--resnet_act_fn", type=str, default="swish", help="activation function in resnets",)
    parser.add_argument("--final_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--attention_posenc", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_connections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--guidance_objective", choices=["none", "solver"], default="none")
    parser.add_argument("--guidance_method", type=str.lower, choices=["none", "tfg"], default="none")
    parser.add_argument("--guidance_params", type=str2dict, default={})
    parser.add_argument("--project_starting", type=int, default=-1)
    parser.add_argument("--project_interval", type=int, default=10)
    parser.add_argument("--snr_loss_weighting", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--base_config", type=str, help="base config to initialize solver with for guidance")
    parser.add_argument("--fem_path", type=str, default="FEM3D/fem3D", help="for solver guidance")
    parser.add_argument("--inverse_target", type=float, help="target for solver loss")
    parser.add_argument("--n_solvers", type=int, default=None)
    parser.add_argument("--material_normalization_path", type=str, default="material_normalization.pkl",
                        help="only for list materials. where to take the (normalization) data from.",
                        )
    parser.add_argument("--no_eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema_beta", type=float, default=0)
    parser.add_argument("--ema_power", type=float, default=2 / 3)
    parser.add_argument("--density_loss_factor", type=float, default=0)
    parser.add_argument("--num_spatial_samples", type=int, default=10)
    parser.add_argument("--reuse_grad_count", type=int, default=1)
    parser.add_argument("--aligned_solvers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--solver_type", choices=["direct", "iterative"], default="direct")
    parser.add_argument("--solver_rel_tol", type=float, default=1e-8)
    parser.add_argument("--solver_abs_tol", type=float, default=-1)
    parser.add_argument("--clip_pred_x0", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--rand_dir_prefix", type=str)
    parser.add_argument("--save_samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save_model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learn", type=str, default="mat", choices=["mat"])

    config = NotADict(**vars(parser.parse_args()))
    if len(sweep_dict) > 0:
        print("updating parameters from sweep dict", sweep_dict)
        config.update(sweep_dict)

    gp = config.guidance_params

    for k, v in config.items():
        glconf[k] = v  # set global config

    if config["guidance_objective"] == "solver":
        if config["aligned_solvers"]:
            if config["n_solvers"] is not None:
                assert (
                    config["n_solvers"] == config["val_batch_size"]
                ), f"n solvers and val batch size must be aligned!"
            else:
                config["n_solvers"] = config["val_batch_size"]
        else:
            if config["n_solvers"] is None:
                config["n_solvers"] = 4  # default value

    assert not (
        config.rescale_betas and config.prediction == "epsilon"
    ), f"Cannot have both epsilon prediction and β rescaling to 0 terminal SNR"
    torch.manual_seed(config["seed"])

    config["orig_name"] = config["name"]
    if config["append_seed_name"]:
        assert config["name"] is not None
        config["name"] += f'_s{config["seed"]}'

    if config["wandb"]:
        wandb.init(name=config.get("name"), config=config)
        print("WANDB run dir", wandb.run.dir)
        results_folder = os.path.join(wandb.run.dir, "results")
    else:
        results_folder = "./results"
    if config["runs_dir"] is not None:
        if config["name"] is not None:
            results_folder = os.path.join(config["runs_dir"], config["name"])
        elif config["rand_dir_prefix"] is not None:
            results_folder = tempfile.mkdtemp(
                prefix=config["rand_dir_prefix"] + "_", dir=config["runs_dir"]
            )
        else:
            raise ValueError("something must be given")
    if config["dir"] is not None:
        if config["runs_dir"] is not None:
            print("Warning: overriding results dir to", config["dir"])
        results_folder = config["dir"]
    print("results folder:", results_folder)
    config["results_folder"] = results_folder

    for k in ["SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID"]:
        if k in os.environ:
            config[k] = os.environ[k]

    # DATASET
    device = "cuda" if th.cuda.is_available() else "cpu"
    if not th.cuda.is_available():
        print("----------- WARNING! RUNNING WITHOUT CUDA!!! -----------")
    ds = get_dataset(config["ds"], config["material_normalization_path"])

    #### DS phase 1, load and check
    x, c, padding = get_x_and_cond(ds, learn=config["learn"])
    ds["padding"] = padding
    N, d_x, *spatial_size = x.shape
    x = th.from_numpy(x)
    if config["conditional"]:
        N_, d_c = c.shape
        assert N == N_, f"got different sample counts {N}, {N_}"
        cond_kde = [gaussian_kde(c[:, 0, None].T)]
        c = th.from_numpy(c)
    else:
        d_c = 0
        cond_kde = None
    spatial_size = tuple(spatial_size)
    if "_valonly" in config["ds"]:
        print("---- interpreting dataset as val_only!  ----")
        assert config["load"], f"cannot train on val_only"
        train_indices = np.array([], dtype=int)
    else:
        split_rng = np.random.default_rng(42)
        indices = split_rng.permutation(N)
        train_fraction = 1.0
        train_indices = indices[: int(N * (train_fraction))]

    #### Model
    downsample_in_mid = False
    downsample_around = False
    if config["downsample"]:
        downsample_in_mid = False
        downsample_around = True

    scalar_cond = (
        ({"K": 1, "K+density": 2}[config["cond_input"]]) if config["conditional"] else 0
    )

    model = MatUNet3DModel(
        sample_size=spatial_size,
        channels=d_x,
        cond_channels=0,  # scalar only now
        cond_in_extra_net=config["cond_net"],
        scalar_cond=scalar_cond,
        downsample_around=downsample_around,
        downsample_in_mid=downsample_in_mid,
        downsample_type=config["downsample_type"],
        time_embedding_type=config["time_embedding"],
        ignore_dims=config["ignore_dims"],
        down_block_types=("DownBlock3D",) * len(config["hidden_sizes"]),
        up_block_types=("UpBlock3D",) * len(config["hidden_sizes"]),
        num_outer_layers=config["num_outer_layers"],
        hidden_sizes=config["hidden_sizes"],
        mid_hidden_sizes=config["mid_hidden_sizes"],
        act_fn=config["resnet_act_fn"],
        final_residual=config["final_residual"],
        skip_connections=config["skip_connections"],
        add_pe_for_attention=config["attention_posenc"],
        add_pe_elsewhere=False,
    )
    model.to(device)

    #### DS phase 2, determine normalization and build datasets
    if not config["load"]:  # train
        x_train = x[train_indices].to(device, th.float32)
        if config["conditional"]:
            c_train = c[train_indices].to(device, th.float32)
            train_ds = TensorDataset(x_train, c_train)
        else:
            train_ds = TensorDataset(x_train)
    val_ds = None

    sched_args = {
        "clip_sample": config["clip_pred_x0"],
        "clip_sample_min": -1.0,
        "clip_sample_max": 1.0,
        "prediction_type": config["prediction"],
    }

    if config["beta_min"] is not None:
        sched_args["beta_start"] = config["beta_min"]
    if config["beta_max"] is not None:
        sched_args["beta_end"] = config["beta_max"]
    sched_args["beta_schedule"] = config["beta_schedule"]
    # https://arxiv.org/pdf/2305.08891
    sched_args["rescale_betas_zero_snr"] = config["rescale_betas"]
    sched_args["timestep_spacing"] = config["timestep_spacing"]
    sched_args["ds"] = ds
    guidance_config = {
        "learn": config["learn"],
        "guidance_params": config["guidance_params"],
        "density_loss_factor": config["density_loss_factor"],
        "reuse_grad_count": config["reuse_grad_count"],
        "aligned_solvers": config["aligned_solvers"],
        "project_starting": config["project_starting"],
        "project_interval": config["project_interval"],
        "objective" : config["guidance_objective"],
        "method" : config["guidance_method"],
    }
    if config["guidance_objective"] == "solver":  # need to talk to solver
        assert config["base_config"] is not None, "need base config to fire up solver!"
        n_el_per_dim = [q - 1 for q in ds["const_n_nodes_per_dim"]]
        # ok cool, now let's instantiate N solvers
        interfaces = [
            SolverInterface(config, ds, n_el_per_dim)
            for _ in range(config["n_solvers"])
        ]
        solver_interface.INTERFACES = interfaces

    sched_args["guidance_config"] = guidance_config

    pipline_args = {
        k: config[k]
        for k in ["guidance_scale", "guidance_rescale", "ema_beta", "ema_power"]
    }

    sched: GuidanceDDPMScheduler = GuidanceDDPMScheduler(
        num_train_timesteps=1000, **sched_args
    )  # always use 1000 for training, but can be less during inference
    pipeline: NDConditionalDDPMPipeline = NDConditionalDDPMPipeline(
        model, sched, **pipline_args
    )

    if config["load"]:
        save_dict = th.load(config["load"], weights_only=True)
        model.load_state_dict(save_dict["model"])
        if "ema" in save_dict:
            pipeline.ema.load_state_dict(save_dict["ema"])
        else:
            print(
                "WARNING: Could not load EMA state from save! overriding ema_model with model!"
            )
            pipeline.ema.ema_model.load_state_dict(save_dict["model"])

    if config["results_folder"] is not None:
        os.makedirs(config["results_folder"], exist_ok=True)
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)

    if config["load"]:  ### eval
        res_acc, res_per, res_acc_ds, plotfig = evaluate(
            config,
            val_ds,
            ds,
            pipeline,
            cond_kde,
            seed=config["eval_seed"],
        )
        print("eval res:", res_acc)
        with open(results_folder + "/eval_res_acc.json", "w") as f:
            json.dump(res_acc, f)
        with open(results_folder + "/eval_res_per.json", "w") as f:
            json.dump(res_per, f)
        if plotfig is not None:
            plotfig.savefig(results_folder + "/samples.png", dpi=200)
            # plt.show()

    else:
        res_acc = train_loop(
            config, train_ds, val_ds, ds, model, sched, pipeline, conditional=cond_kde
        )
        if config["save_model"]:
            th.save(
                model.state_dict(),
                os.path.join(config["results_folder"], "trained_model.pt"),
            )

    if solver_interface.INTERFACES is not None:
        for interface in solver_interface.INTERFACES:
            interface.cleanup()

    return res_acc


if __name__ == "__main__":
    run(sweep_dict={})
