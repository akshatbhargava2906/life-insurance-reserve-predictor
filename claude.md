# Project context

Reserve predictor for the health claims dataset `Reserve_Health_claims_synthetic.csv` (3,000 rows, single line of business). Target column is `Reserve`, heavily right-skewed — modeling happens on `log1p(Reserve)`, predictions get converted back with `expm1`.

See README.md for the current pipeline stage and the findings already established from EDA (driver hierarchy, which categorical fields are statistically significant, which columns get dropped during cleaning).

# Working conventions for this repo

- Never edit or write files directly in this repo. Always hand back complete code in the chat response and let the user paste it in and run it themselves.
- Never run git commit or git push in this repo.
- Scripts hardcode input/output paths as constants near the top of the file (e.g. `INPUT_PATH`, `OUTDIR`) instead of using argparse/CLI arguments.
- Code style: minimal comments, no decorative separator lines (no `"="*70` or `"-"*70` banners), write it like a human wrote it rather than generated boilerplate.
- Outliers in `Reserve` are not data errors — they're real tail-risk claims (concentrated in high-cost benefit categories like ICU, Brain and nervous system). They get flagged (`is_outlier` column), not removed.