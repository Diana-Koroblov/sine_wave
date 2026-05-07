# Sections 2-8 Compliance Checklist

| Section | Status | Notes |
|---|---|---|
| 2 | Pass | SDK boundary, Ruff, pytest coverage, config centralization, docs, notebook, and `uv` tooling are in place. |
| 3 | Partial | Generator and training are correct for the current sigma-aware 15-feature contract, not the older 14-feature wording. |
| 4 | Pass | README plus report material cover setup, analysis, figures, resource usage, and metadata. |
| 5 | Pass | `pyproject.toml` and `uv.lock` are present and used as the environment source of truth. |
| 6 | Pass | Training time and peak RAM are tracked and documented. |
| 7 | Pass | Notebook analysis, evaluation, sensitivity plots, and frequency comparison artifacts exist. |
| 8 | Pass | High-frequency experiment assets and analysis are committed under `assets/v2_high_freq/`. |

## Important Caveat
The current project intentionally keeps the sigma-aware dataset contract because reverting to the older 14-feature variant would be disruptive to correctness and existing tests.
