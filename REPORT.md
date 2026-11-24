# Group Assignment Report

- **Team** (Blackboard group name):
- **Section**: [BBADBA 5A | BBADBA 5B | PPLEDBA 5A]
- **App URL** (on Render):
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
- Buyer type (city council, regional government, ministry, other public body) was added after an experiment
  decided on the validation year: validation MAE 2.308 without it, 2.288 with it (gain 0.020 bidders, 95%
  interval 0.010 to 0.030; `models/experiment_buyer_type.json`). In training it is derived from the buyer's
  name and TED's buyer code; the buyer's name itself is not kept.
- Awards without a call for competition are excluded: nobody can bid for them.
- Time split: settings chosen by training on 2018–2021 (57,114) and validating on 2022 (19,580); final models
  refitted on 2018–2022 (76,694) and tested once on 2023 (19,786). `scripts/train.py` rebuilds everything;
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
| Model, median (used by the app) | 2.239 | 52.5% |
| Model, Poisson mean | 2.480 | 44.4% |

The 80% range covers 90.2% of test cases; its lower bound is always 1 bidder. (An earlier version trained on
2018–2021 only, without buyer type, scored 2.323 against a baseline of 2.442.)

**Component 1, agent** (`uv run python main.py evaluate` → `eval/metrics.json`, `eval/extraction_results.csv`):
field accuracy against TED's official fields on 40 notices of `eval/extraction_truth.csv`.

| Field | Accuracy | n |
|---|---|---|
| *to fill from `eval/metrics.json` once the evaluation has run with a Gemini key* | | |

**TEAM TO WRITE:** what these numbers mean for the product; which fields the agent gets wrong and why it matters
(or not) for the decision; the hand-off cost (model error with the agent's fields vs with the official fields).

### End to end

**TEAM TO WRITE.** How you wrote `eval/cases.csv` (counts by kind, two examples). Results against the two
baselines (`baseline` column, and the median-per-CPV-division rule) from `eval/results.csv`. The judge's rubric
is in `bidsense/judge.py`; report your agreement with it on your own labels (Cohen's kappa in `eval/metrics.json`).

### Cost and latency

*To fill from `eval/metrics.json` → `cost_per_request`: tokens in and out, euros, seconds.* The cost is an
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
  `eval/reference.csv`), with PSI on four features: language (Spanish or not), CPV division, log of the
  value, text length.
- **Alert rule** (`config.py`): a feature's PSI above 0.2 *and* above twice the PSI that chance alone gives for
  the sample size; or more than 30% of texts not in Spanish; or more than 40% 👎.
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
  reading of the buyer may not match them.
- Name removal is rule-based: in the tender text, a name in free prose, without a label or a title, is not
  caught. What the agent writes back (evidence quotes, notes, tool calls) goes through a stricter filter that
  also removes common given names followed by surnames, before it is cached or logged.
- Candidate texts are rebuilt from the notice's fields, so they are cleaner than a real tender PDF.

## Decision

**TEAM TO WRITE.** Which cases go to a person: the rule (`config.py`: value above `REVIEW_VALUE_EUR`, a key
field missing, confidence below `MIN_CONFIDENCE`, expected value within `CLOSE_CALL_EUR` of zero) and how many
cases a day that would be. Go / no-go / go with conditions: the conditions, and the evidence behind them.

## Bonus: reinforcement learning

*Delete this section if you did not do it.* What exists in the code: an epsilon-greedy bandit
(`bidsense/bandit.py`, off by default: `BANDIT_ENABLED` in `config.py`) that chooses between two extraction
prompts (`extract-v1`, `extract-v2`); reward 1 for 👍 and 0 for 👎 on the request that used the prompt; the
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
  the agent's output, and the `used_in` column of the candidate files.

Claude also proposed, and the team accepted or should review: one row per notice with the median offers across
lots as the target; joining contract notices for duration and estimated value; excluding awards without a call
for competition; the small-sample correction in the drift alert; the temporary defaults in `config.py`.

Gemini is part of the product (the agent and the judge). It did not write test cases or labels.

**The team's own work (TEAM TO WRITE, and to confirm):** the product idea and pipeline design, the choice of the
median model for the decision, every threshold marked TEAM TO SET, `eval/cases.csv`, `eval/human_labels.csv`,
the analysis in this report and the go/no-go decision.
