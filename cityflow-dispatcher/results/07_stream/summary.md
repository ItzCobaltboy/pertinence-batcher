# 07 stream analysis

Command: `07_stream_analysis.py`

Inputs:
- models/dispatcher/head.pt
- S02 and S06 videos

Outputs:
- traces/*.csv.gz
- routing_distribution.csv
- stickiness.csv
- transitions.csv
- arrivals.csv
- smoothing.csv
- rolling_share_*.png
- run_hist_*.png

Stickiness (ALL cameras):

| scenario | camera | kind | frames | mean_run_frames | median_run_frames | p90_run_frames | mean_run_sec | median_run_sec | p90_run_sec | switches_per_min |
|---|---|---|---|---|---|---|---|---|---|---|
| S02 | ALL | pred | 4055 | 15.72 | 3.0 | 42.30000000000001 | 3.14 | 0.6 | 8.46 | 18.79 |
| S02 | ALL | ideal | 4055 | 2.57 | 1.0 | 5.0 | 0.51 | 0.2 | 1.0 | 116.6 |
| S06 | ALL | pred | 6006 | 18.04 | 2.0 | 20.80000000000001 | 3.61 | 0.4 | 4.16 | 16.33 |
| S06 | ALL | ideal | 6006 | 2.93 | 1.0 | 5.0 | 0.59 | 0.2 | 1.0 | 102.05 |

Smoothing:

| scenario | window | switches_per_min | mean_run_sec | alpha_sys |
|---|---|---|---|---|
| S02 | 1 | 18.79 | 3.14 | 0.837 |
| S02 | 3 | 8.43 | 6.87 | 0.8395 |
| S02 | 5 | 5.47 | 10.4 | 0.8382 |
| S02 | 10 | 3.11 | 17.63 | 0.8372 |
| S06 | 1 | 16.33 | 3.61 | 0.5997 |
| S06 | 3 | 8.39 | 6.9 | 0.5994 |
| S06 | 5 | 5.24 | 10.82 | 0.5991 |
| S06 | 10 | 3.55 | 15.6 | 0.5992 |

Key numbers:

| scenario | camera | frames | routed_n | routed_s | routed_m | routed_l | ideal_n | ideal_s | ideal_m | ideal_l | alpha_sys |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S02 | ALL | 4055 | 0.003 | 0.7 | 0.002 | 0.294 | 0.519 | 0.32 | 0.104 | 0.056 | 0.837 |
| S06 | ALL | 6006 | 0.707 | 0.169 | 0.102 | 0.023 | 0.56 | 0.124 | 0.12 | 0.196 | 0.5997 |

Flags:
- Stream rate is 5 fps (every 2nd frame of the 10 fps videos), not the stride-5 frames used for training.
- Arrival rates are per camera on video-relative time; the scenario ALL row sums cameras on the same video-relative clock (CityFlow camera start offsets are ignored).
- Smoothing is a causal trailing majority vote, ties go to the larger model.
