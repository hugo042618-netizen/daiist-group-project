# Candidates for the team's test files

Real TED notices collected by `scripts/fetch_candidates.py` (TED Search API, published in 2026).
**They are material to choose from, not test cases.** The team writes `eval/cases.csv` and
`eval/human_labels.csv` by hand; nothing here is copied into them automatically.

| File | What it holds | Use it for |
|---|---|---|
| `typical.csv` | 40 Spanish award notices: open procedure, one lot, €100k-2M | **already used**: `main.py evaluate` measures the agent on them, and the prompt was changed after that measurement |
| `typical_fresh.csv` | 40 more notices defined like `typical.csv`, collected on 25 Nov 2026 and never read by the agent | `normal` cases |
| `unusual.csv` | 25 Spanish award notices: restricted / negotiated / competitive dialogue, frameworks, many lots | `hard` cases |
| `very_large.csv` | 15 Spanish award notices above €10M | `review` cases (value above `REVIEW_VALUE_EUR`) |
| `not_a_tender.csv` | 20 Spanish notices that are not calls for tenders (prior information, contract modifications) | `out_of_scope` cases; add texts of your own (news, an email with no personal data...) |
| `reference_pool.csv` | 45 Spanish notices defined like `typical.csv` (open, one lot, €100k-2M), none of them in the files above | already copied to `eval/reference.csv`; swap rows if you wish |
| `shifted_pool.csv` | 10 Portuguese and 10 French notices | already copied to `eval/shifted.csv` |

**`used_in` tells you which notices are already taken.** All 45 notices of `reference_pool.csv` are marked
`reference.csv`, all 20 of `shifted_pool.csv` are marked `shifted.csv`, and the 40 of `typical.csv` are marked
`extraction_truth.csv (agent measured on it)`: do not use them in `eval/cases.csv`.

**`eval/case_picker.xlsx` is the easy way to choose.** It lists every candidate that is not used yet, one per
row, with four empty columns (kind, expected, baseline, notes) and a helper column that says what the decision
rule gives with the real number of offers. Fill it in, then
`uv run --with openpyxl python scripts/cases_from_picker.py --write` writes `eval/cases.csv` from it
(`scripts/build_case_picker.py` rebuilds the sheet and keeps what you typed).
After you edit `cases.csv`, `reference.csv` or `shifted.csv`, run
`uv run python scripts/fetch_candidates.py --annotate` to refresh the column (it fetches no new notices), and
`main.py evaluate` warns if a case is also a reference input.

Columns: `text` is what you would paste into the app. The other columns are TED's official fields for
the same notice (`cpv_code`, `contract_type`, `estimated_value_eur`, `duration_months`, `procedure`,
`lots_number`, `price_weight_pct`, `is_framework_agreement`, `region`, `buyer_type`) and `offers`, the number of
tenders actually received (the median across lots). `url` opens the notice on TED.

How `text` is built: the API returns each notice as separate fields, not as a document. The text is
the notice's own title and description, word for word, followed by its tender conditions as labelled
lines (value, duration, procedure, lots, award criteria, place). The CPV code and the type of contract
are left out on purpose, so the agent has to work them out. The results section (winners, offers
received) and all contact details are never included, and every text goes through the app's redaction.
For harder cases, paste the full text of a real tender from its PDF (with personal data removed).

Writing `eval/cases.csv` (`id,input,kind,expected,baseline,notes`):

- `input`: a `text` from here, or your own.
- `expected`: one of `BID`, `DONT_BID`, `REVIEW`, `OUT_OF_SCOPE`. Decide it yourselves, with the default
  company settings in `config.py` (bid cost, margin) and the real `offers` as the truth for competition.
- `baseline`: what the company would do today without the product, in the same four words.
- Do not reuse a notice that is in `eval/reference.csv`.

Writing `eval/human_labels.csv` (`id,label,comment`): for 20+ case ids, `label` is `1` if the app's
reasons match the tender and `0` if they do not. Run the app on the case first, then label what it said.

`eval/extraction_truth.csv` holds the official fields of the 80 tender candidates; `main.py evaluate`
compares the agent's extraction with it.
