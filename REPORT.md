# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render): <https://bidsense-ira5.onrender.com>
- **Final commit** (tag `final`):

*At most about 3,000 words, tables excluded. Members and who did what go in CONTRIBUTIONS.md.*

> **Skeleton.** The facts below (what was built, where the data comes from, the numbers already measured)
> are filled in. Everything marked **TEAM TO WRITE** is the team's: the business case, the choice of
> thresholds, the test cases, the labels, the analysis and the decision. Delete this note and the
> TEAM TO WRITE markers before submitting.

## Business case

**TEAM TO WRITE.** Who uses BidSense (the bid manager of a Spanish SME that sells to the public sector?),
which decision it supports (bid or not on one tender), what happens today without it (the baseline: bid on
everything in our sector? a rule of thumb per CPV division?), the success metric, and a rough value in euros
with its assumptions (bids prepared per year × cost per bid × share of hopeless bids avoided...).

## Architecture

```
tender text (pasted)
  └─ redaction: emails, phones, ids, contact names removed        bidsense/redact.py
      └─ COMPONENT 1, agentic LLM: Gemini (gemini-3.5-flash-lite)  bidsense/agent.py
         tools: lookup_cpv · parse_amount · market_stats
         output: 13 fields, each with confidence and evidence
          └─ COMPONENT 2, tabular ML: scikit-learn                  bidsense/model.py
             input: only the agent's fields → expected bidders (median) and an 80% range
              └─ decision and review rule                            bidsense/decision.py
                 win chance = 1 / expected bidders
                 expected value = win chance × contract value × margin − bid cost
                 → BID / DON'T BID / SEND TO REVIEW (or OUT OF SCOPE if it is not a tender)
```

- The agent's output is the model's only input: the model has no access to the text.
- The code checks the agent: CPV codes must exist in the official list, categories must be valid, and an
  evidence quote that is not in the text caps that field's confidence at 0.5.
- If the text is not a tender, the agent says so and the model is skipped.
- Models and APIs: Gemini `gemini-3.5-flash-lite` (free tier) through function calling; four
  `HistGradientBoostingRegressor` models (6 MB in total), which fit in Render's 512 MB.
- Why `gemini-3.5-flash-lite` and not the larger Flash models (team decision, 24 Nov 2026): the app is graded
  by running it with the professor's own free key, so it must work within the free tier. There, `gemini-3.8-flash`
  and `gemini-3.6-flash` allow only 20 requests a day per project (we hit the limit on 24 Nov 2026: quota
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier`, value 20), and one tender needs 3 to 5 requests: four
  tenders a day. Flash-lite allows 500 requests a day and 15 a minute (see "Cost and latency"). It answers in about a second per call, and on the first
  notices we compared it read the same values as `gemini-3.6-flash`. `gemini-3.1-flash-lite` (same limits) is the first fallback and the two
  larger models come after it; the model that answered is logged with every request.

**TEAM TO WRITE:** why these two components and this connection; what was reused from assignments 1–3, and whose.

## Data

| What | Source | Size | Licence |
|---|---|---|---|
| Training data for the bidders model | [TED CSV open data](https://data.europa.eu/data/datasets/ted-csv), contract award notices and contract notices 2018–2023, Spain only | 122,301 Spanish award notices; 96,480 usable (87.8% of those open to competition report a valid number of offers) | CC BY 4.0 |
| CPV list for `lookup_cpv` | CPV 2008, Publications Office of the EU ([eForms SDK](https://github.com/OP-TED/eForms-SDK)) | 9,454 codes, Spanish and English labels | EU open data |
| Candidate test notices | [TED Search API](https://docs.ted.europa.eu/api/latest/search.html), notices published in 2026 | 165 notices in `eval/candidates/` | CC BY 4.0 |

- Target: `NUMBER_OFFERS` ("number of tenders received"), the median across the lots of a notice.
- Features, all of them fields the agent extracts: CPV division and group, contract type, procedure, region,
  buyer type, estimated value (log), duration, number of lots, weight of price, framework agreement. Duration
  and the estimated value come from the contract notice linked to each award; the awarded price is never used.
- Buyer type (city council or other local authority, regional government, ministry, other public body) was
  added after an experiment decided on the validation year: validation MAE 2.308 without it, 2.284 with it (gain
  0.024 bidders, 95% interval 0.014 to 0.034; `models/experiment_buyer_type.json`). The classes are those of
  TED's code list "buyer legal type", the same definition the agent's prompt uses. In training the class is
  derived from the buyer's name and TED's older buyer code; the buyer's name itself is not kept.
- The buyer-type rule changed once, on 25 Nov 2026: provincial and island councils (diputaciones, cabildos,
  consells: 4.6% of the training notices) moved from `regional_government` to `city_council`, so that training
  and the prompt agree, and the models were retrained (`bidders-hgb-v3`). With the earlier rule the experiment
  gave 2.308 and 2.288, and the test-year MAE of the median model was 2.239.
- Awards without a call for competition are excluded: nobody can bid for them.
- Time split: settings chosen by training on 2018–2021 (57,114) and validating on 2022 (19,580); final models
  refitted on 2018–2022 (76,694) and tested on 2023 (19,786), twice: before and after the change of the
  buyer-type rule described above. `scripts/train.py` rebuilds everything;
  raw TED files are never committed.
- Personal data: the training table only reads structured columns (no names, addresses or ids). Candidate
  texts hold the notice's title, description and conditions; the results section and contact details are
  never included, and every text goes through the redaction. Logs store a summary of the input (CPV division,
  value bucket, language, text length), never the text.

**TEAM TO WRITE:** confirm, after reading your final `eval/` files and a sample of the logs, that no personal data is in them.

## Evaluation

### Components

**Component 2, bidders model** (test year 2023, 19,786 notices; `models/metrics.json`):

| | MAE (bidders) | Bucket accuracy (1–2 / 3–5 / 6+) |
|---|---|---|
| Baseline: median offers per CPV division | 2.419 | 48.5% |
| Model, median (used by the app) | 2.243 | 52.4% |
| Model, Poisson mean | 2.461 | 44.8% |

The 80% range covers 91.0% of test cases; its lower bound is always 1 bidder. (An earlier version trained on
2018–2021 only, without buyer type, scored 2.323 against a baseline of 2.442.)

**Component 1, agent** (`uv run python main.py evaluate` → `eval/metrics.json`, `eval/extraction_results.csv`):
field accuracy against TED's official fields on 40 notices of `eval/extraction_truth.csv`.

All 40 notices were read by `gemini-3.5-flash-lite`, the app's default model, and by no other
(`eval/extraction_results.csv`, column `model`). **There are two measurements, because the prompt changed
after the first one.** On 24 Nov 2026 the agent was measured with prompt `extract-v1`. On 25 Nov the
definition of buyer type in the prompt was rewritten from TED's code list "buyer legal type" (prompt
`extract-v3`; nothing else in the prompt changed, and the 40 answers were not read one by one to write it),
and the same 40 notices were read again. They are therefore no longer unseen notices for the current prompt,
and they are marked as used in `eval/candidates/` so that no test case comes from them.

| Field | `extract-v1` (24 Nov) | `extract-v3` (25 Nov, current) | n |
|---|---|---|---|
| CPV division (2 digits) | 75% | 75% | 40 |
| CPV group (3 digits) | 60% | 57% | 40 |
| Contract type | 95% | 98% | 40 |
| Estimated value (within 1%) | 100% | 100% | 40 |
| Duration in months (within 1) | 100% | 100% | 37 |
| Procedure | 100% | 100% | 40 |
| Number of lots | 98% | 100% | 40 |
| Weight of price (within 2 points) | 100% | 100% | 37 |
| Framework agreement | 100% | 100% | 40 |
| Region | 100% | 100% | 40 |
| Buyer type | 64% | 62% | 39 |

- One notice is 2.5 points, so every difference between the two columns is a single notice. The new
  definition of buyer type did not raise the agreement with TED.
- Buyer type, current prompt: 24 of 39 agree. Of the 15 differences, 11 are buyers that TED files as a
  regional authority and the agent called "other", 2 the other way round, and 2 are local authorities the
  agent called "other". The agent's confidence was 0.84 on average when it agreed and 0.81 when it did not.
  The truth for this field is the legal type the buyer declared in its TED notice, not a label written by a
  person.
- Recognised as a tender: 39 of 40 in both measurements. The one it refuses is an award notice, which the
  agent says is not a call for tenders.
- CPV: the test texts do not state the CPV code, so the agent has to choose it from the description with
  `lookup_cpv`.
- Hand-off (40 notices from 2026 whose real number of offers is known, current prompt and current model): MAE
  of the expected bidders 1.720 with the agent's fields, 1.675 with TED's official fields, 1.788 for the
  baseline. Small sample. (First measurement, with the earlier model: 1.756, 1.651 and 1.788.)
- The first measurement is kept in `eval/extraction_results_v1.csv`.

**TEAM TO WRITE:** what these numbers mean for the product; which fields the agent gets wrong and why it matters
(or not) for the decision; the hand-off cost (model error with the agent's fields vs with the official fields).

### End to end

**TEAM TO WRITE.** How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the two
baselines (`baseline` column, and the median-per-CPV-division rule) from `eval/results.csv`. The judge's rubric
is in `bidsense/judge.py`; report your agreement with it on your own labels (Cohen's kappa in `eval/metrics.json`).

### Cost and latency

*To fill from `eval/metrics.json` → `cost_per_request` once `eval/cases.csv` exists.* Measured so far on the
40 extraction notices with `gemini-3.5-flash-lite` (25 Nov 2026): 3.5 requests, 10,011 input tokens and 508
output tokens per tender on average, €0.0039 per tender; the agent took 3.6 s (median) and 5.0 s (mean) per
tender, retries included. The cost is an
estimate at Gemini's paid-tier prices for the model that answered (`config.py`: for `gemini-3.5-flash-lite`,
$0.30 per million input tokens and $2.50 per million output tokens); on the free tier the app costs nothing,
but it is limited as follows. The scikit-learn model costs nothing per request.

**Free-tier limits** (per project, as shown in AI Studio on 24 Nov 2026; `config.py`, `FREE_TIER_LIMITS`):

| Model | Role in the app | Requests per minute | Requests per day | Tenders per day (3 to 5 requests each) |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | default | 15 | 500 | about 100 to 160 |
| `gemini-3.1-flash-lite` | first fallback | 15 | 500 | about 100 to 160 |
| `gemini-3.8-flash` | later fallback | 5 | 20 | about 4 to 6 |
| `gemini-3.6-flash` | later fallback | 5 | 20 | about 4 to 6 |

- Daily quotas reset at midnight Pacific time (09:00 in Madrid). On 24 Nov 2026 we used all 500 daily requests
  of `gemini-3.5-flash-lite` and all 20 of each larger model while building and testing.
- The app sends at most 80% of a model's per-minute limit (12 requests a minute on the lite models), moves to
  the next model when one is busy or out of quota, and says so in the answer when none is available.
- Billing was added to the project on the evening of 24 Nov 2026, so the cache was filled on the paid tier
  from then on: €0.18 to finish the first reading (167 requests), and €0.47 to read all 105 inputs again with
  the current prompt on 25 Nov (420 requests to `gemini-3.5-flash-lite`, 1,238,743 input and 54,011 output
  tokens). What was read earlier on the free tier was not charged. The app's defaults stay those of the free
  tier.
- The evaluation inputs are answered from the committed cache (`cache/`), so `check.py`, `main.py evaluate` and
  `main.py drift-demo` do not use the quota of whoever runs them; `cache/agent_models.csv` records which model
  answered each input.

**TEAM TO WRITE:** cost per month at the volume of your business case, and whether the free tier is enough for
it (the limits above cap the app at roughly 100 tenders a day).

## Operations

- **Logged per request** (Render Postgres online, `logs/` locally): input summary, the drift features, the
  agent's JSON (fields, confidence, evidence, tool calls), the prediction, the decision and its reasons, model
  and prompt versions, latency, tokens in and out, estimated cost, and the 👍/👎 with its optional comment.
- **Quality signals over time:** share sent to review, share of 👎 among rated requests, mean confidence
  (rolling average of the last 10 requests).
- **Drift check:** the last 100 requests against `monitoring/reference_features.csv` (the 45 inputs of
  `eval/reference.csv`), with PSI on four features: language in scope or not, CPV division, log of the
  value, text length.
- **Languages in scope:** Spanish, Catalan, Galician and Basque. Spanish tenders are also published in the
  co-official languages, so a text in Catalan from a company in Catalonia is normal traffic and must not set
  off the alert; Portuguese, French or English texts are out of scope. The language is detected in code
  (`bidsense/lang.py`, stop words), not by the agent, so the check works even when Gemini is down. All 45
  reference inputs are in scope.
- **Alert rule** (`config.py`): a feature's PSI above 0.2 *and* above twice the PSI that chance alone gives for
  the sample size; or more than 30% of texts in a language out of scope; or more than 40% 👎.
- **Drift demo:** `eval/shifted.csv` holds 10 Portuguese and 10 French notices.

**TEAM TO WRITE:** the justification of each threshold, the drift-demo result with numbers
(`uv run python main.py drift-demo`), and the rate of false alarms on normal inputs.

## Failures

**TEAM TO WRITE.** Failed end-to-end cases grouped by cause (agent, model, or the hand-off), with counts, and
what you would fix first. Known limits to consider:

- The TED CSV stops in 2023 and the median number of offers fell from 3 (2018–2021) to 2 (2022–2023): the
  model is three years behind the tenders it will see.
- The model beats the baseline by 0.18 bidders (about 7%): a real but modest gain.
- Buyer type adds only 0.02 bidders, and that was measured with labels derived from TED data; the agent's own
  reading of the buyer may not match them. The test year was also read twice (see Data), so the second
  figure is not a first look at unseen data; it moved by 0.004 bidders.
- Name removal is rule-based: in the tender text, a name in free prose, without a label or a title, is not
  caught. What the agent writes back (evidence quotes, notes, tool calls) goes through a stricter filter that
  also removes common given names followed by surnames, before it is cached or logged.
- Candidate texts are rebuilt from the notice's fields, so they are cleaner than a real tender PDF.

## Decision

**Company defaults, set by the team on 25 Nov 2026** (`config.py`; each company can change both in the app):

| Setting | Value | Source |
|---|---|---|
| Cost of preparing one bid | €11,400 | Ecorys and Capgemini Invent for the European Commission, [*Cost-benefit analysis on the quality and efficiency of public procurement procedures*](https://public-buyers-community.ec.europa.eu/resources/evaluation-public-procurement-directives-external-studies) (2025), section 5.2: €11,400 per bidder and procedure above the EU thresholds, 2019–2024 (€10,900 in open procedures) |
| Margin on the contract value | 6% | Art. 131 of the general regulation of the public contracts law ([Real Decreto 1098/2001](https://www.boe.es/buscar/act.php?id=BOE-A-2001-19995)): 6% of the budget as the contractor's industrial profit |

- About the sources: the €11,400 is an EU average from a survey, with no figure for Spain. The earlier study for
  the Commission (PwC, London Economics and Ecorys, [*Public procurement in Europe: cost and effectiveness*](https://op.europa.eu/en/publication-detail/-/publication/0cfa3445-7724-4af5-8c2b-d657cd690c03),
  2011, Figure 2.12, p. 88) gave €3,000 per bid for a firm in Spain and €3,800 on average in Europe, at 2009
  wages. Article 131 sets the 6% for the budgets of works contracts; using it for supplies and services is the
  team's assumption.
- Why not €3,000: with it the rule never says DON'T BID on tenders of the size TED publishes (table below).

**Sensitivity to the bid cost.** What the rule answers when it is told TED's official fields and the real
number of offers, on the 77 unused tender candidates with a known number of offers (`eval/case_picker.xlsx`;
values from €245,000 up, margin 6%, close-call band €2,000, review above €5 million;
`scripts/build_case_picker.py --mix 3000 11400`):

| Bid cost | BID | DON'T BID | REVIEW | BID when value ÷ bidders is above | DON'T BID when it is below |
|---|---|---|---|---|---|
| €3,000 | 46 | 0 | 31 | €83,333 | €16,667 |
| €11,400 (default) | 33 | 14 | 30 | €223,333 | €156,667 |

In both rows 24 of the reviews are contracts above €5 million; the others are close calls (7 and 6).

**TEAM TO WRITE.** Which cases go to a person: the rule (`config.py`: value above `REVIEW_VALUE_EUR`, a key
field missing, confidence below `MIN_CONFIDENCE`, expected value within `CLOSE_CALL_EUR` of zero) and how many
cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.

## Bonus: reinforcement learning

*Delete this section if you did not do it.* What exists in the code: an epsilon-greedy bandit
(`bidsense/bandit.py`, off by default: `BANDIT_ENABLED` in `config.py`) that chooses between two extraction
prompts (`extract-v3`, and `extract-v4`, the same with step-by-step instructions); reward 1 for 👍 and 0 for 👎 on the request that used the prompt; the
policy is rebuilt from the logs before each request; a simulated warm-up (`scripts/simulate_bandit.py`) is
stored apart in `monitoring/bandit_warmup.json` and shown in its own column in the Monitoring tab.

**TEAM TO WRITE:** the justification of the reward and what it could reward by accident, why epsilon-greedy and
what exploring costs, the evidence from real logs (and which part was simulated), the risks.

## How to run it

See "How to run" at the top of README.md. Only `GEMINI_API_KEY` is needed; no download step. First start
takes a few seconds; a request takes a few seconds on the free tier (instant when cached).

## AI use

Claude Code (Anthropic's coding assistant) wrote the code of this repository under the team's direction, in
the order of the commits on the `bidsense` branch:

- data preparation and model training scripts (`scripts/ted_data.py`, `scripts/train.py`), after reading the TED codebook;
- the agent, its tools, prompts and redaction (`bidsense/`), the pipeline and the decision rule as specified by the team;
- the Gradio interface, the logging additions, the monitoring and drift check, the evaluation script and the LLM-judge rubric;
- the script that collects candidate notices from the TED API (`scripts/fetch_candidates.py`), and it filled
  `eval/reference.csv` and `eval/shifted.csv` from those candidates;
- the bandit module and its simulator; the "How to run" section of the README and the skeleton of this report;
- at the team's request: the final refit on 2018–2022, the buyer-type experiment, the stricter name filter on
  the agent's output, and the `used_in` column of the candidate files;
- at the team's request, on 25 Nov 2026: the buyer-type definition of the prompt rewritten from TED's code list,
  and `eval/case_picker.xlsx` (the unused candidates with empty columns for the team) with the script that
  copies the filled sheet into `eval/cases.csv`. Claude also collected 40 more candidate notices
  (`typical_fresh.csv`), because the first 40 had been used to measure the agent.

Claude also proposed, and the team accepted or should review: one row per notice with the median offers across
lots as the target; joining contract notices for duration and estimated value; excluding awards without a call
for competition; the small-sample correction in the drift alert; the temporary defaults in `config.py`.

Gemini is part of the product (the agent and the judge). It did not write test cases or labels.

**The team's own work (TEAM TO WRITE, and to confirm):** the product idea and pipeline design, the choice of the
median model for the decision, every threshold marked TEAM TO SET, `eval/cases.csv`, `eval/human_labels.csv`,
the analysis in this report and the go/no-go decision.
