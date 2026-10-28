# PERTINENCE NSGA-II dispatcher on CityFlow, v1 pool (current fine-tuned n/s/m)

Method: original vendored PERTINENCE search (13 genes, pop 50, 50 gens, penalty-matrix loss, INS/ISNS/ENS), 2,500 evaluations, front over all evaluations. Run on CPU in the sandbox VM; the FC trainer is a numpy port of the torch one (same maths, `$HOME/shim/fc_numpy.py`, not in the repo yet). Final numbers on S02, scored once. Dispatcher cost includes extractor 3.156 GFLOPs.

| protocol | head trained on | NSGA fitness on | front pts | above static hull | best gain over hull |
|---|---|---|---|---|---|
| s05 (primary) | S05 cameras fold A (7,729 fr) | S05 fold B (7,046 fr) | 109 | 8 | +0.013 alpha_sys at 146 GFLOPs (84% l) |
| train (old protocol) | S01/S03/S04 (88% n labels) | S05 | 119 | 28 | +0.038 alpha_sys at 21.8 GFLOPs (50% n, 50% s) |

Static on S02: n 0.518 @ 8.7, s 0.780 @ 28.6, m 0.867 @ 78.9, l 1.0 @ 165.1, oracle 1.0 @ 32.1 GFLOPs.
No front point is balanced (max share <= 0.7, min share >= 0.05) in either protocol.

Reading: in the primary protocol the dispatcher is at best on par with static policies. The train-protocol gain sits in the cheap n/s region and uses a mismatched label distribution, treat with care. Caveat: v1 fine-tune had the bias-lr spike, v2 not yet run (needs the laptop GPU).
