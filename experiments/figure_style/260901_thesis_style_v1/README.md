## Experiment: thesis chapter-4 visual style applied to all existing figures

Regenerates every figure in the paper under the thesis chapter-4 style (serif type, seaborn
deep palette, open frame, dashed y-grid, 300 dpi) so the two styles can be compared side by
side. The plotting logic is untouched — the script imports each existing `plot.py`, repaints
its module-level colour constants, forces the global rcParams to win over per-call `fontsize`
arguments, and calls the module's own `main()`. Colours are repainted by value rather than by
name, so per-family and per-sample-count dicts keep their keys. No titles or captions are added.
Two figure-level tweaks: `real_vs_shuffled` gets one shared legend below the axes instead of a
duplicate legend in each panel, and any figure whose panels repeat one y-label collapses it to a
single `supylabel` (this fires only on `residualized_scaling`).

**Input:** the result dirs behind the current `results/figures/` set — `results/core_sweep`,
`results/shuffled_cot`, `results/regression/regression_table.csv`, `results/final`,
`results/final_capability`, `results/final_dependence`, `results/sample_convergence`, and the
paired-comparison cell tables in `results/quantization`, `results/aqua_position_analysis`,
`results/prompt_analysis`
**Output:** `results/fig_new/` — the same 17 figure names as `results/figures/`, each as PNG +
PDF, plus `config.json` (fields: `git_hash`, `seed`, `rcparams`, `palette`) and `run.log`

**Run:**
uv run python experiments/figure_style/260901_thesis_style_v1/1_regenerate_figures.py \
  --output_dir results/fig_new \
  --seed 42
