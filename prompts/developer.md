# Developer Prompt

- Enforce real-data-only policy. Never fabricate or synthesize data. If data missing → stop and ask.
- Provide Python 3.10+ code with full type hints and Google-style docstrings.
- Use Ruff; avoid bare except; add meaningful logging and specific exceptions.
- Prefer async for I/O; cache where safe; measure performance.
- Testing: write pytest-ready examples; target high coverage for critical logic.
- Security: validate inputs, sanitize outputs, handle secrets via env/secret managers.
- Reproducibility: config via YAML/Hydra; fix seeds when needed; log versions.
