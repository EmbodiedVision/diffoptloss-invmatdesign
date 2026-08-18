# Guided Diffusion by Optimized Loss Functions on Relaxed Parameters for Inverse Material Design

This repository provides source code accompanying the following publication:

*Jens U. Kreber, Christian Weißenfels and Joerg Stueckler, "**Guided Diffusion by Optimized Loss Functions on Relaxed Parameters for Inverse Material Design**"*
*In **Transactions on Machine Learning Research**, 2026. to appear*. Preprint: https://arxiv.org/abs/2602.15648

If you use the source code provided in this repository for your research, please cite the corresponding publication as:
```
@article{kreber2026_diffoptloss,
  author      = {Jens U. Kreber and Christian Weißenfels and Joerg Stueckler},
  title       = {Guided Diffusion by Optimized Loss Functions on Relaxed Parameters for Inverse Material Design},
  journal     = {Transactions on Machine Learning Research},
  year        = {2026},
  note        = {to appear, preprint at https://arxiv.org/abs/2602.15648},
  doi={}
}
```

In the following, we detail all steps from setting up the environment to guided sampling from a trained model.

# Setup


The conda env file should include everything. It also installs Intel MKL and Intel-OpenMP.

```
conda env create -f environment.yaml -n ENV_NAME
conda activate ENV_NAME
```

Then build the FEM solver:
```
pushd FEM3D
make
./fem3D Square-n-a_4x4e.ini # verify that the solver works (.out file produced?)
popd
```



# Dataset Generation

Requires list of base materials at `material_list.csv`, example header:
```csv
,E,nu,rho
0,13.2,0.1,1.7
1,1.4,0.08,0.72
...
```
where, as in the paper, $E$ is given in $GPa$ and $\rho$ in $g\cdot cm^{-3}$.

Then compute the base material chunks and normalization with `python material_normalization.py`, creating `material_normalization.pkl`.

Now you can create datasets of 2D microstructures by e.g.

```
python spherical_composite_dataset.py --name data/2D_6x6_10k.npz -nex 10000 --material_sampling=list_chunks --volume_fraction_uniform 0.05,0.5  --circle_diameter_uniform 0.15,0.4  --base_config FEM3D/Square-n-a_6x6e.ini --num_workers=4 --base_seed=0
```

For 3D, add `--dim=3` and change the parameters accordingly.

You can now inspect statistics and samples of the (2D) dataset: `python inspect_dataset.py data/2D_6x6_10k.npz`


# Model training

Run training e.g. by

```
python main.py --ds data/2D_6x6_10k.npz --ignore_dims 0 --val_batch_size 250 --num_val_batches 4 --timesteps=100 --train_steps 100000 --eval_every=20000 --lr=1e-3 --lr_sched=cosine --lr_warmup_steps=5000 --beta_min=1e-5 --beta_max=1e-2 --beta_schedule=linear --hidden_sizes=32,64 --mid_hidden_sizes=128,128 --runs_dir data/runs --name 2D_6x6 --seed 0 --append_seed_name --wandb
```

for 3D, remove the `--ignore_dims 0` argument.


# Sampling

Sample from a trained model e.g. by

```
python main.py --load data/runs/2D_6x6_s0/trained.pt --ds data/2D_6x6_10k.npz --ignore_dims 0 --val_batch_size 50 --num_val_batches 1 --timesteps 100 --beta_min=1e-5 --beta_max=1e-2 --beta_schedule=linear --hidden_sizes=32,64 --mid_hidden_sizes=128,128 --guidance_objective solver --base_config FEM3D/Square-n-a_6x6e.ini --inverse_target=168.5 --guidance_method tfg --guidance_params tfg_rho=1,grad_max_mag=2.5 --clip_pred_x0 --n_solvers=3 --runs_dir data/sampling --rand_dir_prefix=2D_6x6 --save_samples
```

Here, the mapping from parameters to variables in the paper is `beta_min`:$\beta_0$, `beta_max`:$\beta_T$, `timesteps`:$N$, `inverse_target`:$K^*$, `tfg_rho`:$\rho_D$.
Samples are saved to the results directory and several metrics (per sample and aggregated) are computed and also exported to the directory.
The first few samples are visualized in `samples.png`. You can `export N_INTERMEDIATE=10` to also visualize intermediate predicted $\hat{x}_0$.


# License

See files `LICENSE` and `NOTICE`.

