## Experiment: thesis chapter-4 visual style applied to all existing figures

Regenerates every figure in the paper under the thesis chapter-4 style (serif type, seaborn
deep palette, open frame, dashed y-grid, 300 dpi) so the two styles can be compared side by
side. The plotting logic is untouched — the script imports each existing `plot.py`, repaints
its module-level colour constants, forces the global rcParams to win over per-call `fontsize`
arguments, and calls the module's own `main()`. No titles or captions are added.

**Input:** the existing result dirs — `results/core_sweep`, `results/shuffled_cot`,
`results/regression/regression_table.csv`, `results/robustness`, `results/capability`,
`results/cot_dependence`
**Output:** `results/fig_new/` — 12 figures as PNG + PDF, plus `config.json`
(fields: `git_hash`, `seed`, `rcparams`, `palette`) and `run.log`

**Run:**
uv run python experiments/figure_style/260901_thesis_style_v1/1_regenerate_figures.py \
  --output_dir results/fig_new \
  --seed 42
