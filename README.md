# cerebras-pytorch-src

This repository is a clean-room compatibility scaffold for the public
`cerebras.pytorch` API documented by Cerebras. It is not a verbatim copy of the
closed-source `cerebras_pytorch` wheel and does not implement Cerebras WSC/CSX
hardware execution internals.

The current goal is to provide import-compatible CPU/GPU behavior for common
helpers used by downstream projects such as model code, tests, and local
experiments.

## Layout

- `src/cerebras/pytorch`: public `cerebras.pytorch` package surface.
- `src/cerebras_pytorch`: legacy import alias.
- `tests`: smoke tests for import compatibility and lightweight behavior.

## Development

The project is configured for UV:

```bash
uv sync --extra dev
uv run pytest
```

The local shell used to create this scaffold did not have `uv` or `torch`
installed, so the modules are defensive on import. Tensor and optimizer
features require installing the project dependencies.
