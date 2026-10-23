# Routing summary: run `all`

Main point: **80** (highest final_val alpha_sys): final_val alpha_sys 0.7455, cost 149.18 GFLOPs.
Dispatcher on every frame (D = 1). Routing behaviour only; no correctness on CityFlow.

| scenario | cams | frames | share n | share s | share m | share l | mean run n (frames) | mean run s (frames) | mean run m (frames) | mean run l (frames) | switches/min | max 10 s share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S01 | 5 | 10281 | 0.0% | 0.0% | 25.1% | 74.9% | - | - | 15.9 | 46.6 | 18.79 | 79.8% |
| S02 | 4 | 8109 | 0.0% | 0.0% | 59.2% | 40.8% | - | - | 16.9 | 11.7 | 41.66 | 71.2% |
| S03 | 6 | 13517 | 0.0% | 0.0% | 6.5% | 93.5% | - | - | 8.4 | 113.9 | 9.00 | 100.0% |
| S04 | 25 | 10780 | 0.0% | 0.0% | 9.8% | 90.2% | - | - | 8.9 | 71.5 | 12.80 | 100.0% |
| S06 | 6 | 12006 | 0.0% | 0.0% | 65.4% | 34.6% | - | - | 24.0 | 12.8 | 32.23 | 82.0% |

Kill criterion (heaviest-loaded model above 90:10):
- S01: heaviest yolov8l at 74.9% of frames; above 90:10: no
- S02: heaviest yolov8m at 59.2% of frames; above 90:10: no
- S03: heaviest yolov8l at 93.5% of frames; above 90:10: yes
- S04: heaviest yolov8l at 90.2% of frames; above 90:10: yes
- S06: heaviest yolov8m at 65.4% of frames; above 90:10: no
- all scenarios: heaviest yolov8l at 68.6% of frames; above 90:10: no
