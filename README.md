# H-GVT: Hierarchical Graph VQ-Transformer

Two-stage molecular graph generation on QM9, ZINC250k, MOSES and GuacaMol:

1. **Stage 1** (`src/graph_var.py`): spectrally-regularized multi-scale VQ-VAE tokenizer (GVAR).
2. **Stage 1.5** (`src/gen_var_latents.py`): encode the training set into discrete latent codes (H5, with the tokenizer weights embedded).
3. **Stage 2** (`src/train_ar.py`): decoder-only Transformer trained autoregressively over the coarse-to-fine codes.
4. **Sampling & evaluation** (`src/test_sample.py`): AR sampling -> VQ-VAE decoding -> RDKit validation -> Validity/Uniqueness/FCD/NSPDK (QM9, ZINC250k), MOSES metrics, or the GuacaMol distribution-learning benchmark.

## Installation

```bash
uv sync
source .venv/bin/activate
```

## Pretrained checkpoints

```
checkpoints/
  QM9/       vqvae.pt  latents.h5  ar.pt
  ZINC/      vqvae.pt  latents.h5  ar.pt
  MOSES/     vqvae.pt  latents.h5  ar.pt
  GuacaMol/  vqvae.pt  latents.h5  ar.pt
```

| Dataset  | VQ-VAE (epoch)                                   | AR (epoch)                          |
|----------|--------------------------------------------------|-------------------------------------|
| QM9      | codebook 1024, scales [8,4,2,1] (300)            | 8L / 8H / 256d (200)                |
| ZINC250k | codebook 1024, scales [8,4,2,1] (2000)           | 8L / 8H / 256d (200)                |
| MOSES    | codebook 4096, scales [16,8,4,2,1] (2000)        | 12L / 16H / 512d (400)              |
| GuacaMol | codebook 4096, scales [32,16,8,4,2,1] (1900)     | 12L / 16H / 640d (400)              |

`latents.h5` embeds the same VQ-VAE weights as `vqvae.pt`, so sampling only needs `ar.pt` + `latents.h5`; the AR checkpoint already references the matching `latents.h5` path.

## Sampling / evaluation

Run from the repository root (datasets are downloaded automatically to `data/` on first use):

```bash
python src/test_sample.py ar_ckpt=checkpoints/QM9/ar.pt       total_samples=10000
python src/test_sample.py ar_ckpt=checkpoints/ZINC/ar.pt      total_samples=10000
python src/test_sample.py ar_ckpt=checkpoints/MOSES/ar.pt     total_samples=10000
python src/test_sample.py ar_ckpt=checkpoints/GuacaMol/ar.pt  total_samples=10000
```

Extra sampling knobs: `temperature`, `top_k`, `do_sample`, `batch_size`, `max_new_tokens` (see `src/config/sample_ar.yaml`).

## Training from scratch

All commands run from the repository root.

### Stage 1: VQ-VAE tokenizer

```bash
# QM9 (default config)
accelerate launch --config_file acc.yaml src/graph_var.py

# ZINC250k
accelerate launch --config_file acc.yaml src/graph_var.py dataset.name=ZINC \
    train.epochs=2000 train.bs=128 \
    gvar.encoder.node_feat_dim=9 gvar.encoder.output_dim=48 \
    gvar.decoder.input_dim=48 gvar.decoder.output_dim=9

# MOSES
accelerate launch --config_file acc.yaml src/graph_var.py dataset.name=MOSES \
    train.epochs=2000 train.bs=512 \
    gvar.cluster_sizes=[16,8,4,2,1] \
    gvar.lamb_commit=0.2 gvar.lamb_ent=0.1 gvar.lamb_link=0.1 \
    gvar.encoder.node_feat_dim=7 gvar.encoder.hidden_dim=192 gvar.encoder.output_dim=64 \
    gvar.vq.codebook_size=4096 \
    gvar.decoder.input_dim=64 gvar.decoder.hidden_dim=192 gvar.decoder.output_dim=7

# GuacaMol
accelerate launch --config_file acc.yaml src/graph_var.py dataset.name=Guacamol \
    train.epochs=2000 train.bs=90 train.ckpt_freq=20 \
    gvar.cluster_sizes=[32,16,8,4,2,1] \
    gvar.lamb_commit=0.2 gvar.lamb_ent=0.1 gvar.lamb_link=0.1 \
    gvar.encoder.node_feat_dim=12 gvar.encoder.hidden_dim=192 gvar.encoder.output_dim=64 \
    gvar.vq.codebook_size=4096 \
    gvar.decoder.input_dim=64 gvar.decoder.hidden_dim=192 gvar.decoder.output_dim=12
```

Checkpoints are written to `outputs/vqvae/<timestamp>/checkpoints/`.

### Stage 1.5: latent codes

```bash
python src/gen_var_latents.py \
    var_ckpt_path=outputs/vqvae/<timestamp>/checkpoints/gvar_final.pt \
    save_path=checkpoints/QM9/latents.h5
```

### Stage 2: AR Transformer

```bash
# QM9
accelerate launch --config_file acc.yaml src/train_ar.py \
    dataset.save_path=checkpoints/QM9/latents.h5 train.epochs=200 train.bs=512

# ZINC250k
accelerate launch --config_file acc.yaml src/train_ar.py \
    dataset.save_path=checkpoints/ZINC/latents.h5 train.epochs=200 train.bs=128

# MOSES
accelerate launch --config_file acc.yaml src/train_ar.py \
    dataset.save_path=checkpoints/MOSES/latents.h5 train.epochs=400 train.bs=512 \
    model.gpt.n_layer=12 model.gpt.n_head=16 model.gpt.n_embd=512

# GuacaMol
accelerate launch --config_file acc.yaml src/train_ar.py \
    dataset.save_path=checkpoints/GuacaMol/latents.h5 train.epochs=400 train.bs=128 \
    model.gpt.n_layer=12 model.gpt.n_head=16 model.gpt.n_embd=640
```

Checkpoints are written to `outputs/ar/<timestamp>/checkpoints/`.
