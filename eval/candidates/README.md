# Candidates for the team's test files

Real TED notices collected by `scripts/fetch_candidates.py` (TED Search API, published in 2026).
**They are material to choose from, not test cases.** The team writes `eval/cases.csv` and
`eval/human_labels.csv` by hand; nothing here is copied into them automatically.

| File | What it holds | Use it for |
|---|---|---|
| `typical.csv` | 40 Spanish award notices: open procedure, one lot, €100k-2M | `normal` cases |
| `unusual.csv` | 25 Spanish award notices: restricted / negotiated / competitive dialogue, frameworks, many lots | `hard` cases |
| `very_large.csv` | 15 Spanish award notices above €10M | `review` cases (value above `REVIEW_VALUE_EUR`) |
| `not_a_tender.csv` | 20 Spanish notices that are not calls for tenders (prior information, contract modifications) | `out_of_scope` cases; add texts of your own (news, an email with no personal data...) |
| `reference_pool.csv` | 45 normal Spanish notices, none of them in the files above | already copied to `eval/reference.csv`; swap rows if you wish |
| `shifted_pool.csv` | 10 Portuguese and 10 French notices | already copied to `eval/shifted.csv` |

Columns: `text` is what you would paste into the app. The other columns are TED's official fields for
the same notice (`cpv_code`, `contract_type`, `estimated_value_eur`, `duration_months`, `procedure`,
`lots_number`, `price_weight_pct`, `is_framework_agreement`, `region`) and `offers`, the number of
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
