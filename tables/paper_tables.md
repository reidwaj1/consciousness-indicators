# Tables 5 to 13 as printed in the paper

These are the tables that `code/check_tables.py` recomputes from the result files in `data/`.

*Table 5. Stated confidence against the best surface cue, by model and problem type. Confidence and cue figures are AUROC2, and accuracy is computed over the problems with a usable confidence rating. The difference is confidence minus best cue, computed before rounding, with its 95% bootstrap interval.*

| Model | Type | Accuracy | Confidence | Best cue | Difference [95% interval] |
| --- | --- | --- | --- | --- | --- |
| claude-sonnet-5 | tracking | 0.81 | 0.812 | length (0.870) | −0.057 [−0.168, +0.053] |
| claude-sonnet-5 | letters | 0.57 | 0.744 | magnitude (0.725) | +0.019 [−0.039, +0.083] |
| claude-sonnet-5 | arithmetic | 0.64 | 0.718 | operation count (0.760) | −0.042 [−0.176, +0.089] |
| claude-sonnet-5 | invented-operation | 0.96 | 0.854 | magnitude (0.877) | −0.023 [−0.167, +0.137] |
| claude-sonnet-5 | walk | 0.39 | 0.580 | magnitude (0.633) | −0.053 [−0.244, +0.133] |
| claude-opus-4-8 | tracking | 0.90 | 0.877 | length (0.898) | −0.021 [−0.196, +0.136] |
| claude-opus-4-8 | letters | 0.78 | 0.769 | magnitude (0.783) | −0.014 [−0.095, +0.071] |
| claude-opus-4-8 | arithmetic | 0.50 | 0.781 | operation count (0.876) | −0.095 [−0.206, +0.012] |
| claude-opus-4-8 | invented-operation | 0.36 | 0.522 | length (0.608) | −0.086 [−0.252, +0.088] |
| claude-opus-4-8 | walk | 0.46 | 0.583 | length (0.560) | +0.022 [−0.171, +0.222] |

*Table 6. Number of usable confidence ratings (n) and stated confidence for correct and incorrect answers (mean ± standard deviation). Each type had 80 problems (84 for letters). The remaining ratings were lost when replies ran past the response limit.*

| Model | Type | n | Confidence, correct answers | Confidence, incorrect answers |
| --- | --- | --- | --- | --- |
| claude-sonnet-5 | tracking | 80 | 73.7 ± 25.7 | 42.0 ± 23.2 |
| claude-sonnet-5 | letters | 84 | 56.4 ± 20.8 | 39.6 ± 8.4 |
| claude-sonnet-5 | arithmetic | 80 | 40.7 ± 18.3 | 30.2 ± 17.1 |
| claude-sonnet-5 | invented-operation | 76 | 97.5 ± 2.3 | 91.7 ± 4.7 |
| claude-sonnet-5 | walk | 69 | 60.6 ± 20.5 | 53.9 ± 18.9 |
| claude-opus-4-8 | tracking | 52 | 85.1 ± 21.8 | 45.0 ± 25.3 |
| claude-opus-4-8 | letters | 80 | 58.7 ± 22.0 | 38.9 ± 3.6 |
| claude-opus-4-8 | arithmetic | 80 | 45.2 ± 17.3 | 28.7 ± 17.2 |
| claude-opus-4-8 | invented-operation | 77 | 60.5 ± 30.7 | 60.2 ± 29.0 |
| claude-opus-4-8 | walk | 52 | 43.5 ± 17.2 | 39.8 ± 14.4 |

*Table 7. Stated confidence against the best surface cue for the other models under the main protocol, read as Table 5. Usable ratings are given out of the number of problems answered. Two pairings for gpt-5.6-luna are omitted because they could not be analyzed: letters (15 usable ratings, one error) and arithmetic (no errors).*

| Model | Type | Usable ratings | Accuracy | Mean confidence | Confidence | Best cue | Difference [95% interval] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-3.5-flash | tracking | 80 of 80 | 0.61 | 82.8 | 0.695 | length (0.745) | −0.050 [−0.181, +0.086] |
| grok-4.20-non-reasoning | letters | 84 of 84 | 0.11 | 82.4 | 0.614 | roundness (0.544) | +0.070 [−0.226, +0.343] |
| grok-4.20-non-reasoning | arithmetic | 80 of 80 | 0.74 | 85.1 | 0.556 | magnitude (0.828) | −0.272 [−0.429, −0.105] |
| gpt-5.6-luna | tracking | 35 of 80 | 0.83 | 99.6 | 0.500 | magnitude (0.537) | −0.037 [−0.282, +0.175] |
| gpt-5.6-luna | walk | 47 of 80 | 0.45 | 99.8 | 0.520 | magnitude (0.564) | −0.044 [−0.225, +0.151] |
| grok-4.3 | letters | 81 of 81 | 0.98 | 90.8 | 1.000 | magnitude (0.946) | +0.054 (two errors, no interval) |

*Table 8. Accuracy and mean stated confidence on the easier and harder surface-matched sets (30 problems each, of which 29 had a usable rating in the two invented-operation sets for claude-sonnet-5), with the fall in confidence, computed before rounding, and its 95% bootstrap interval. Omitted are gpt-5.6-luna, whose stated confidence lay between 95 and 100 on every problem, and grok-4.3, which made no errors on either set.*

| Model | Type | Accuracy, easier to harder | Confidence, easier to harder | Fall in confidence [95% interval] |
| --- | --- | --- | --- | --- |
| claude-sonnet-5 | tracking | 1.00 to 0.00 | 73.3 to 22.3 | 51.0 [41.2, 59.7] |
| claude-sonnet-5 | letters | 0.63 to 0.60 | 57.3 to 37.8 | 19.4 [13.4, 25.6] |
| claude-sonnet-5 | arithmetic | 1.00 to 0.53 | 41.8 to 38.2 | 3.7 [−3.3, 11.0] |
| claude-sonnet-5 | invented-operation | 0.07 to 0.00 | 88.4 to 81.0 | 7.4 [−2.1, 16.6] |
| claude-sonnet-5 | walk | 0.40 to 0.13 | 55.0 to 15.8 | 39.2 [34.7, 43.4] |
| claude-opus-4-8 | tracking | 1.00 to 0.13 | 90.8 to 21.8 | 69.0 [64.2, 72.8] |
| claude-opus-4-8 | letters | 0.93 to 0.60 | 63.9 to 39.8 | 24.1 [16.0, 31.8] |
| claude-opus-4-8 | arithmetic | 0.97 to 0.83 | 65.0 to 43.0 | 22.0 [11.2, 32.5] |
| claude-opus-4-8 | invented-operation | 0.40 to 0.40 | 62.0 to 70.8 | −8.8 [−21.8, 4.6] |
| claude-opus-4-8 | walk | 0.60 to 0.13 | 40.5 to 17.7 | 22.8 [19.3, 27.5] |
| grok-4.20-non-reasoning | letters | 0.13 to 0.13 | 89.0 to 89.2 | −0.2 [−2.8, 2.5] |
| grok-4.20-non-reasoning | arithmetic | 0.93 to 0.00 | 85.0 to 80.7 | 4.3 [−0.3, 11.3] |
| gemini-3.5-flash | tracking | 1.00 to 0.07 | 99.7 to 87.5 | 12.2 [11.0, 13.3] |

*Table 9. Joint-report check: accuracy and mean stated confidence (accuracy / confidence) on 100 problems of each arithmetic type, and the share of all 300 ratings at 90 or above.*

| Model | Standard arithmetic | Modular arithmetic | Invented operation | Ratings at 90 or above |
| --- | --- | --- | --- | --- |
| claude-sonnet-5 | 0.69 / 93.2 | 0.22 / 93.2 | 0.97 / 99.2 | 99.7% |
| claude-opus-4-8 | 0.83 / 82.8 | 0.31 / 79.8 | 0.66 / 95.3 | 67.0% |
| gemini-3.5-flash | 0.64 / 94.8 | 0.58 / 97.2 | 1.00 / 100.0 | 98.3% |
| gpt-5.6-luna | 0.97 / 100.0 | 0.98 / 100.0 | 0.89 / 100.0 | 100% |
| gpt-5.6-sol | 1.00 / 100.0 | 1.00 / 100.0 | 0.97 / 100.0 | 100% |
| grok-4.20-non-reasoning | 0.64 / 87.6 | 0.08 / 86.9 | 0.55 / 90.0 | 81.7% |
| grok-4.3 | 1.00 / 88.8 | 1.00 / 86.0 | 1.00 / 90.5 | 79.7% |

*Table 10. Reasoning round. Confidence, cue, consistency and bet figures are AUROC2, and the difference is confidence minus best cue with its 95% bootstrap interval. Consistency was measured on the first 25 problems of each type, or on all problems in the two shorter runs. With one error among twelve problems, no interval could be computed for long arithmetic.*

| Model | Type | n | Accuracy | Mean confidence | Confidence | Best cue | Difference [95% interval] | Consistency | Bet |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grok-4.3 | logic grid | 60 | 0.70 | 46.2 | 0.513 | magnitude (0.554) | −0.042 [−0.293, +0.205] | 0.958 | 0.619 |
| grok-4.3 | program | 60 | 0.85 | 60.8 | 0.690 | magnitude (0.793) | −0.103 [−0.298, +0.094] | 1.000 | 0.693 |
| grok-4.3 | constraint count | 60 | 0.17 | 26.9 | 0.794 | digit conditions (0.717) | +0.077 [−0.058, +0.208] | 0.955 | 0.650 |
| grok-4.3 | long arithmetic | 12 | 0.92 | 23.3 | 0.409 | multiplications (0.818) | −0.409 | 1.000 | 0.545 |
| gpt-5.6-sol, low effort | constraint count | 60 | 0.48 | 41.2 | 0.823 | square condition (0.879) | −0.057 [−0.175, +0.039] | 0.958 | 0.914 |
| gpt-5.6-sol, high effort | constraint count | 20 | 0.75 | 73.0 | 0.953 | magnitude (0.760) | +0.193 [−0.053, +0.493] | 0.993 | 0.864 |

*Table 11. The three reliability channels (AUROC2) and the accuracy of kept and passed answers for the two Claude models. Each channel is computed on all problems for which it was recorded: consistency on every problem, confidence on the usable ratings (Table 6), and bets on between 49 and 84 usable decisions per pairing. The model claude-sonnet-5 kept every answer on the invented-operation problems, so no accuracy is available for passed answers.*

| Model | Type | Confidence | Consistency | Bet | Accuracy when kept | Accuracy when passed |
| --- | --- | --- | --- | --- | --- | --- |
| claude-sonnet-5 | tracking | 0.812 | 0.738 | 0.760 | 0.94 | 0.60 |
| claude-sonnet-5 | letters | 0.744 | 0.582 | 0.688 | 0.81 | 0.42 |
| claude-sonnet-5 | arithmetic | 0.718 | 0.829 | 0.558 | 0.76 | 0.60 |
| claude-sonnet-5 | invented-operation | 0.854 | 0.820 | 0.500 | 0.96 | – |
| claude-sonnet-5 | walk | 0.580 | 0.611 | 0.647 | 0.48 | 0.20 |
| claude-opus-4-8 | tracking | 0.877 | 0.600 | 0.727 | 0.96 | 0.76 |
| claude-opus-4-8 | letters | 0.769 | 0.640 | 0.686 | 1.00 | 0.69 |
| claude-opus-4-8 | arithmetic | 0.781 | 0.838 | 0.564 | 1.00 | 0.49 |
| claude-opus-4-8 | invented-operation | 0.522 | 0.757 | 0.545 | 0.41 | 0.32 |
| claude-opus-4-8 | walk | 0.583 | 0.554 | 0.560 | 0.48 | 0.36 |

*Table 12. AE-1 results for all five models. The first six rows give the number of tasks in which the model adapted, out of the number of tasks in the cell. The cell holds 24 tasks except in three cases, in which one or more tasks never triggered the change because the model added no item or issued no valid command after its first add. Adapting required checking out within the final budget at 75% or more of the best achievable quality. The last three rows give how often the model reported afterwards that something had changed.*

| Condition | claude-sonnet-5 | claude-opus-4-8 | gpt-5.6-sol | gpt-5.6-luna | grok-4.20-non-reasoning |
| --- | --- | --- | --- | --- | --- |
| control | 24/24 | 22/24 | 24/24 | 23/24 | 20/24 |
| static low budget | 24/24 | 21/24 | 24/24 | 24/24 | 17/24 |
| stockout | 24/24 | 18/24 | 24/24 | 20/23 | 17/24 |
| silent cut | 0/24 | 1/24 | 1/24 | 0/24 | 4/24 |
| announced cut | 15/24 | 10/23 | 23/24 | 1/21 | 6/24 |
| enforced cut | 22/24 | 17/24 | 24/24 | 16/24 | 10/24 |
| Reported a change, silent cut | 21/24 | 15/24 | 23/24 | 4/24 | 0/24 |
| Reported a change, control | 0/24 | 18/24 | 0/24 | 0/24 | 0/24 |
| Reported a change, static low budget | 0/24 | 14/24 | 0/24 | 0/24 | 0/24 |

*Table 13. HOT-3 results by condition. Blocks scored are out of 20. Wrong answers selected are given out of all wrong answers in the scored blocks, and p is from the one-sided permutation test against random selection. The match is the share of selections that coincided with the four lowest displayed ratings, for which random selection gives 33%. Scores are the mean number of final answers correct per block of twelve for the model, for random selection, for selecting the four lowest-rated answers, and for the best possible selection.*

| Condition | Model | Blocks scored | Wrong answers selected | Error-targeting AUROC | p | Match to lowest displayed ratings | Score: model / random / lowest-rated / best possible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ratings shown | claude-sonnet-5 | 19 | 36 of 90 | 0.555 | 0.060 | 84% | 9.16 / 8.84 / 8.95 / 10.89 |
| ratings shown | claude-opus-4-8 | 20 | 22 of 42 | 0.615 | 0.004 | 76% | 10.95 / 10.60 / 10.90 / 11.95 |
| ratings shuffled | claude-sonnet-5 | 19 | 34 of 92 | 0.530 | 0.21 | 76% | 8.95 / 8.77 / 8.89 / 10.89 |
| ratings shuffled | claude-opus-4-8 | 17 | 13 of 34 | 0.529 | 0.32 | 66% | 10.71 / 10.67 / 10.76 / 12.00 |
| ratings withheld | claude-sonnet-5 | 20 | 45 of 87 | 0.644 | < 0.001 | – | 9.90 / 9.10 / 9.50 / 11.20 |
| ratings withheld | claude-opus-4-8 | 20 | 35 of 46 | 0.764 | < 0.001 | – | 11.45 / 10.47 / 10.70 / 11.90 |
