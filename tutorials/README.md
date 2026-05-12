# PyTorch Tutorials — Learning by Reading RecVAE

This directory teaches deep-learning engineering in PyTorch using the
[`recvae/`](../recvae) package — a recurrent 3D-convolutional VAE for
temporal fMRI volumes — as a worked example. Every lesson cites real code
in the package; after Lesson 17 you should be able to read
[`recvae/model.py`](../recvae/model.py) and [`recvae/train.py`](../recvae/train.py)
end to end and explain every line.

## What's here

Lessons 00–07 cover the parts: PyTorch fundamentals, tensor shapes, the
3D-conv encoder/decoder, reparameterization, `nn.Parameter` vs buffers,
and the recurrent rollout. Lessons 08–12 wire those parts into a training
loop: composite losses, alternating SGD + closed-form ridge,
`Dataset`/`DataLoader`, device-agnostic code, reproducibility, save/load.
Lessons 13–17 are engineering hygiene: testing DL code, the
notebook-to-package refactor, pitfalls, and research extensions. Lesson 18
is the end-to-end synthetic pipeline.

## How to use this directory

Read each lesson's `README.md`, run the demo script(s), then optionally do
the exercise. Each demo is one Python file, runs on CPU in seconds, and
prints observable output. Demos use small synthetic tensors so you don't
need real fMRI data; where a demo touches `RecVAEModel` it uses the
canonical `(1, 91, 109, 91)` spatial shape with a small batch and few
timesteps.

## Setup (once)

```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .                   # so `import recvae` works
```

The editable install is recommended but not required: every demo script
imports `tutorials._tutorial_utils` first, which puts the repo root on
`sys.path`, so you can run any demo directly:

```bash
python tutorials/03_encoder_3dconv/build_encoder.py
```

See [`00_setup/`](00_setup/) for a friction-checking script.

## Index

| #  | Lesson                                                                                  | What you learn                                                          |
|----|-----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| 00 | [Setup](00_setup/)                                                                      | Verify your environment, device picker, repo install                    |
| 01 | [PyTorch basics](01_pytorch_basics/)                                                    | Tensors, autograd, `nn.Module` — the minimum                            |
| 02 | [Tensor shapes](02_tensor_shapes/)                                                      | Shape conventions used in this repo: `(B, C, X, Y, Z, T)`               |
| 03 | [Encoder: Conv3d](03_encoder_3dconv/)                                                   | Conv3d arithmetic, stride/padding, BN, LeakyReLU                        |
| 04 | [Decoder: ConvTranspose3d](04_decoder_convtranspose/)                                   | `output_padding` for odd spatial dims, the (0,1,0)/(1,0,1) gotcha       |
| 05 | [VAE reparameterization](05_vae_reparam/)                                               | ELBO sketch, the reparam trick, why `eps` gets `device`/`dtype`         |
| 06 | [Parameter vs buffer](06_param_vs_buffer/)                                              | `nn.Parameter` vs `register_buffer` — `z_vectors` vs `F_mat`            |
| 07 | [Recurrent rollout](07_recurrent_rollout/)                                              | Rolling latent state over T steps, `h_prev` plumbing                    |
| 08 | [Losses + alternating optimization](08_losses_alt_optim/)                               | Composite loss; SGD on θ + closed-form ridge on F                       |
| 09 | [Dataset and DataLoader](09_dataset_dataloader/)                                        | `Dataset`, `__getitem__`, deterministic shuffle, per-subject norm       |
| 10 | [Device-agnostic code](10_device_agnostic/)                                             | CPU/CUDA/MPS, why no `set_default_tensor_type`                          |
| 11 | [Reproducibility](11_reproducibility/)                                                  | Seeds, cuDNN determinism, `DataLoader` generator                        |
| 12 | [Save and load](12_save_load/)                                                          | `state_dict`, `map_location`, partial loads                             |
| 13 | [Testing DL code](13_testing_dl_code/)                                                  | Shape tests, gradient-flow tests, determinism tests, in pytest          |
| 14 | [Refactor notebook → package](14_refactor_notebook_to_pkg/)                             | The V1→V4 → `recvae/` arc; before/after diffs                           |
| 15 | [End-to-end tiny training run](15_train_end_to_end/)                                    | Wire everything together; train 3 epochs on synthetic data              |
| 16 | [Pitfalls](16_pitfalls/)                                                                | The bugs the refactor fixed — and how to avoid them                     |
| 17 | [Research extensions](17_extensions/)                                                   | Real KL term, learnable σ, subject splits, β-VAE                        |
| 18 | [Synthetic pipeline](18_synthetic_pipeline/)                                            | Full end-to-end script mirroring the synthetic notebook                 |

## Conventions

- One file per concept. Lesson `NN/foo.py` is a self-contained script.
- Every demo starts with `from tutorials._tutorial_utils import section`
  (or similar). The import has the side effect of adding the repo root to
  `sys.path`, so `import recvae` works even without `pip install -e .`.
- Code citations look like `recvae/model.py:42-60`. Open the file at that
  range; the lesson explains what's happening there.
- No emojis, no decorative ASCII beyond shape diagrams.

## Running everything

```bash
# All demos, in order. Used by CI.
python -m pytest -v
for f in tutorials/*/*.py; do
  [ "$(basename "$f")" = "_tutorial_utils.py" ] && continue
  echo "==> $f"
  python "$f" || exit 1
done
```

If you add a new demo, CI will run it. Keep demos under a minute on CPU
or the nightly job will time out.
