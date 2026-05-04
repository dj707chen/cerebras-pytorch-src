# Cerebras PyTorch source code

Recreate the source code respostory of Cerebras PyTorch.

## What it is

Cerebras PyTorch `cerebras_pytorch` is a closed-source package built and distributed by **Cerebras Systems**.
We want to re-create its source respository.
The package is a component of the Cerebras Wafer-Scale Cluster (WSC) software stack that wraps and extends standard PyTorch to run on Cerebras CS-series hardware (wafer-scale chips) with near-perfect linear scaling across millions of cores — without requiring the user to manage distributed computing themselves.

It also provides general ML helpers that work on CPU/GPU, which is why `cerebras_modelzoo` (this repo) depends on it even for non-hardware runs.

- **PyPI page**: https://pypi.org/project/cerebras-pytorch/
- **Docs**: https://docs.cerebras.net/
- **Homepage**: https://cerebras.net/
- **Support**: support@cerebras.net
- **Discord**: https://discord.gg/ZqvYS2e2rY

## Source code

`cerebras_pytorch` is **not open source**. There is no public GitHub repository for it. The package is distributed as a prebuilt wheel through:

1. **PyPI** (public) — general-purpose CPU/GPU builds:
   ```bash
   pip install cerebras_pytorch
   ```
2. **Cerebras CSoft platform** (private, for WSC hardware users) — hardware-accelerated builds installed as part of the Cerebras software stack on the appliance. This is what the `PYTHON-SETUP.md` in this repo refers to when it says "After installing all the Cerebras packages distributed in the CSoft platform."

But you can find source code from the API doc: https://training-api.cerebras.ai/en/latest/_modules/cerebras/pytorch/backend.html#backend

## How to get source code
- Start from https://training-api.cerebras.ai/en/latest/wsc/api/cerebras_pytorch/index.html,
- navigate the tree,
- download the Python code from each page
- Save the code in this repository, organize them in package hiarchy.

## Project build

Use UV.
