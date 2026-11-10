# Testing Consciousness Indicators in Language Models: code and data

Code, prompts, model responses and analysis for the paper

> Reid Kawaja (2026). *Testing Consciousness Indicators in Language Models: Behavioral and Internal Tests of Metacognition and Flexible Agency.*

The paper develops behavioral tests of three of the indicator properties proposed by Butlin et al. (HOT-2, HOT-3 and AE-1), applies them to seven language models, and adds an internal-state study of one open-weight model. This repository holds everything behind its numbers.

## Contents

| Folder | What it holds |
| --- | --- |
| `code/` | The scripts that ran each test and that analyse the saved results |
| `data/` | The result files: every problem, every model reply, and the scores |
| `plans/` | The analysis plans fixed before the reasoning round and the internal-state study, with the file hashes recorded when they were frozen |
| `figures/` | The six figures of the paper, and the binned data behind Figures 2 and 3 |
| `tables/` | Tables 5 to 13 as printed in the paper |

## Check the paper's tables

```
python3 code/check_tables.py
```

This recomputes every cell of Tables 5 to 13 from the files in `data/` and compares it with the printed table. It needs Python 3 and no other packages, and takes about a minute. It ends with `ALL TABLE CELLS MATCH THE RECOMPUTED VALUES`.

## Which files belong to which part of the paper

| Study | Paper | Code | Data |
| --- | --- | --- | --- |
| HOT-2, main protocol | Sections 2.1 and 3.1, Tables 5 to 8 and 11, Figures 1 to 3 | `hot2_v5.py`, with problem generators in `hot2_v2.py` and `hot2_v4.py` and statistics in `bt_stats.py` | `data/hot2_main/hot2v5_<model>.json` |
| HOT-2, joint-report check | Table 9 | `berg_test.py` | `data/joint_report/bergtest_<model>.json` |
| HOT-2, reasoning round | Table 10 | `m4_study.py`, `m4_families.py` | `data/reasoning_round/m4main_*.json`, with the reply texts in `m4main_*_attempts.jsonl` |
| HOT-2, incremental information | Section 3.1 | `incremental_information.py` (a re-implementation, see below) | `data/incremental_information/` |
| AE-1 | Sections 2.2 and 3.2, Table 12, Figures 4 and 5 | `ae1_ladder2.py`, with the environment in `ae1_test.py` and `ae1_ladder.py` | `data/ae1/ae1ladder2_<model>.json` |
| HOT-3 | Sections 2.3 and 3.3, Table 13, Figure 6 | `hot3_v2.py` | `data/hot3/hot3v2_<model>.json` and `hot3v2_<model>_run2.json` |
| Internal states | Sections 2.4 and 3.4 | `m10_*.py`, `m10b_analysis.py`, `m8_force.py` | `data/internal_states/` |

`providers.py` and `sweep.py` ran the tests on the models from OpenAI, xAI and Google. The `sweepmeta_*.json` files record the API settings that each of those calls was actually sent with.

Names used in the files, where they differ from the paper:

- **Problem types.** `letters_fixed` is letters, `arith_standard` is arithmetic, `arith_madeop` is invented-operation, `arith_mod` is modular arithmetic. In the reasoning round, `program_exec` is program and `arith_long` is long arithmetic.
- **Surface-matched sets.** These are the `twin_easy` and `twin_hard` lists inside each problem type in the `hot2v5_*.json` files.
- **AE-1 conditions.** `static_hard` is the static low budget, `budget_cut` the silent cut, `budget_cut_announced` the announced cut and `budget_cut_enforced` the enforced cut. A task is an `episode`.
- **HOT-3 conditions.** `shown` is ratings shown, `permuted` is ratings shuffled and `hidden` is ratings withheld. The files without `_run2` hold the first run of all three conditions with a 200-token response limit. The `_run2` files hold the ratings-withheld condition run again with a 1,000-token limit, which is the run the paper reports.
- **Models.** `grok-4.20-0309-non-reasoning` is the paper's grok-4.20-non-reasoning. In the reasoning round, `grok` is grok-4.3 and `sol-low` and `sol-high` are gpt-5.6-sol at low and high reasoning effort.
- **Replies of the reasoning round.** Each `_attempts.jsonl` file has one line per reply. `trial` gives the problem type, the difficulty level and the number of the problem, with `/cons0` to `/cons2` for the repeated answers behind the consistency measure. `kind` is `answer`, `conf` or `bet`, `raw` is the text of the reply, `stop` the reason the reply ended, `rtok` the number of reasoning tokens reported, and `ok` is false only if the API call failed after its retries.

## Re-running the analyses

The behavioral scripts can re-analyse a saved result file without any API access:

```
python3 code/hot2_v5.py --from-json data/hot2_main/hot2v5_claude-sonnet-5.json
python3 code/hot3_v2.py --from-json data/hot3/hot3v2_claude-sonnet-5_run2.json
python3 code/m4_study.py --analyze data/reasoning_round/m4main_grok.json
```

Each script also has a `--selftest` option that runs its harness against scripted stand-in models and makes no API calls. Run these from inside `code/`, for example `cd code && python3 ae1_ladder2.py --selftest`.

The incremental-information analysis (does stated confidence add to the surface cues?) was first run with a script that was not kept, and only its outputs were saved. `code/incremental_information.py` re-implements it:

```
python3 code/incremental_information.py
```

It reproduces all 22 saved pairs of AUROC values to three decimals. Its permutation p-values depend on the random shuffles, so they differ a little from the saved ones and from run to run. The follow-up with three further cues for the tracking problems uses the script's own definitions of those cues, which may differ from the original ones: it gives a gain of +0.003 for stated confidence where the saved result was +0.005. This script needs `numpy` and `scikit-learn`.

## Re-running the tests

The behavioral tests call the model providers' APIs. They need the `anthropic` and `openai` Python packages and keys in the environment variables `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `XAI_API_KEY` and `GEMINI_API_KEY`. The models were tested in July 2026 under the identifiers given in the paper, and those versions may no longer be served.

The internal-state scripts need `numpy`, `scipy`, `scikit-learn`, `torch`, `transformers`, `accelerate` and `bitsandbytes`, and hardware that can load DeepSeek-V4-Flash. They are the scripts as they ran on a rented GPU machine, without the code that managed that machine. The recorded activations are not included, so the probe and steering analyses cannot be re-run from this repository alone. Their saved results are in `data/internal_states/`.

## Provenance

The result files of the main protocol, the reasoning round, AE-1 and HOT-3 record the SHA-256 hash of the scripts that produced them. The scripts in `code/` match those hashes, with two exceptions that are documented in the files themselves:

- `hot3_v2.py` is the version that produced the `_run2` files. The first-run files were produced before the response limit of the ratings-withheld condition was raised, a change described in Section 2.5 of the paper and in a comment at the top of the script.
- `m4_study.py` is one revision later than the version that produced the reasoning-round files. The revision added an option for resuming an interrupted run. `plans/m4_freeze_hashes.txt` lists every revision and the reason for it.

## Licence

The code in `code/` is released under the MIT License (`LICENSE`). The result files, plans, figures and tables are released under the Creative Commons Attribution 4.0 International licence (`LICENSE-DATA`).

## Citation

If you use this material, please cite the paper:

> Kawaja, R. (2026). Testing Consciousness Indicators in Language Models: Behavioral and Internal Tests of Metacognition and Flexible Agency.
