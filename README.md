# cerebras-pytorch-src

Recreated source tree for `cerebras.pytorch` from Cerebras public docs viewcode pages.

## Recreate Sources

```bash
uv run python scripts/recreate_from_docs.py
```

This crawls:

- `https://training-api.cerebras.ai/en/latest/wsc/api/cerebras_pytorch/index.html`
- `https://training-api.cerebras.ai/en/latest/_modules/index.html`

Then it downloads all discovered `_modules/cerebras/pytorch/*.html` pages and writes
their extracted Python source into the local `cerebras/pytorch/...` hierarchy.
