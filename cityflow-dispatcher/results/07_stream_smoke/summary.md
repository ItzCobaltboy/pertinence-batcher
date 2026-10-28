# 07 stream analysis

Command: `07_stream_analysis.py --smoke`

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
| S02 | ALL | pred | 200 | 5.56 | 2.0 | 10.5 | 1.11 | 0.4 | 2.1 | 52.5 |
| S02 | ALL | ideal | 200 | 1.87 | 1.0 | 4.0 | 0.37 | 0.2 | 0.8 | 159.0 |

Smoothing:

| scenario | window | switches_per_min | mean_run_sec | alpha_sys |
|---|---|---|---|---|
| S02 | 1 | 52.5 | 1.11 | 0.82 |
| S02 | 3 | 28.5 | 2.0 | 0.84 |
| S02 | 5 | 22.5 | 2.5 | 0.87 |
| S02 | 10 | 4.5 | 10.0 | 0.925 |

Key numbers:

| scenario | camera | frames | routed_n | routed_s | routed_m | routed_l | ideal_n | ideal_s | ideal_m | ideal_l | alpha_sys |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S02 | ALL | 200 | 0.19 | 0.0 | 0.0 | 0.81 | 0.025 | 0.42 | 0.35 | 0.205 | 0.82 |

Flags:
- Stream rate is 5 fps (every 2nd frame of the 10 fps videos), not the stride-5 frames used for training.
- Arrival rates are per camera on video-relative time; the scenario ALL row sums cameras on the same video-relative clock (CityFlow camera start offsets are ignored).
- Smoothing is a causal trailing majority vote, ties go to the larger model.
