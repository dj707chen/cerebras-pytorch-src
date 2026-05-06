# cerebras-pytorch source project

This directory contains Python sources unpacked from the `cerebras-pytorch`
2.10.0 wheel. The added `pyproject.toml` makes it usable as a UV-managed local
project without pulling the wheel runtime dependencies.

## Compile

```bash
uv python install 3.11
uv sync
uv run compile-cerebras
```

Compiled bytecode is written under `build/pyc/` so the unpacked `cerebras/`
tree stays clean.
