# Contributing

FrameSeek is an evidence-first video research runtime. Changes should preserve
three invariants:

1. Every generated citation refers to a frame that the selected backend received.
2. Core indexing and validation remain usable without a model SDK.
3. Cloud transfer is explicit and documented.

## Development setup

```bash
python -m venv .venv
python -m pip install -e .[dev]
python -m unittest discover -s tests -v
ruff check .
```

Open an issue before adding a new model backend or changing the index schema.
Backend pull requests should include protocol tests that do not require live API
credentials. Never commit model weights, videos without redistribution rights,
or API keys.
