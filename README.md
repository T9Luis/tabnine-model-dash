[![CI](https://github.com/T9Luis/tabnine-model-dash/actions/workflows/ci.yml/badge.svg)](https://github.com/T9Luis/tabnine-model-dash/actions/workflows/ci.yml)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tabnine-model-dash.streamlit.app)

# Tabnine Model Dashboard

A live, interactive Streamlit dashboard comparing every AI model available
in Tabnine — covering capability scores, official benchmarks, hardware
requirements, and task-specific recommendations.

## Features

The dashboard ships with six tabs:

**Overview** — ranked bar chart, code-generation-vs-reasoning scatter, and a full capability heatmap across all models.

**Side-by-Side Compare** — radar/spider chart plus grouped bar chart for any selection of up to 8 models chosen in the sidebar.

**Benchmarks** — HumanEval+, MBPP, SWE-bench, GPQA, MMLU, and LiveCodeBench scores, pulled live from public APIs and falling back to static baseline data when APIs are unreachable.

**Task Advisor** — pick a developer task (code completion, debugging, refactoring, etc.) and get ranked model recommendations with a winner callout.

**Hardware** — VRAM, GPU topology, and context-window comparisons for self-hosted models (Devstral, MiniMax M2.7, Laguna XS.2).

**Full Table** — searchable, downloadable CSV of the entire registry.

## Quick Start

```bash
git clone https://github.com/<your-org>/tabnine-model-dash.git
cd tabnine-model-dash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

## Deploying to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in at [share.streamlit.io](https://share.streamlit.io).
3. Click **New app** and point it at `app.py` in this repo.
4. Deploy — no secrets or environment variables are required for the public benchmark APIs.

## Live Data Sources

| Source | Data | Refresh |
|---|---|---|
| [EvalPlus](https://evalplus.github.io/leaderboard.html) | HumanEval+ pass@1 | Hourly (TTL cache) |
| [PapersWithCode](https://paperswithcode.com/sota/software-engineering-on-swe-bench-verified) | SWE-bench resolved % | Hourly |
| [LiveCodeBench](https://livecodebench.github.io/) | Contamination-free coding | Hourly |

When all three APIs are unreachable the dashboard falls back to static
baseline scores from Tabnine's official documentation and published model
cards. A banner at the top of the page indicates whether live data is active.

## Static Data Sources

- [Tabnine AI Models](https://docs.tabnine.com/main/welcome/readme/ai-models)
- [Model Settings](https://docs.tabnine.com/main/administering-tabnine/managing-your-team/settings/models-settings)
- [System Requirements](https://docs.tabnine.com/main/welcome/readme/system-requirements/system-requirements)

## Project Structure

```
tabnine-model-dash/
├── app.py                    # Streamlit entry point
├── requirements.txt
├── .streamlit/
│   └── config.toml           # Tabnine brand theme
├── data/
│   ├── models.py             # Static model registry
│   └── live_benchmarks.py    # Live API fetchers with TTL caching
├── utils/
│   └── dataframe.py          # DataFrame builders and transformations
├── tests/
│   └── test_models.py        # pytest suite
└── .github/
    └── workflows/
        └── ci.yml            # CI — lint + test + smoke test
```

## Running Tests

```bash
pytest tests/ -v
```

## CI

GitHub Actions runs on every push to `main` / `develop` and on pull
requests. It also runs on a daily schedule to catch any breaking changes in
upstream benchmark APIs.
