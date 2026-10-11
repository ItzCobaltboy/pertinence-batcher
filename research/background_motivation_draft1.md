# Background and Motivation (Draft 1)

> Status: working draft, not paper text. Inline `[TODO]` = needs data or work. `[VERIFY]` = claim taken from an abstract or secondhand description; check against the full paper before keeping it. Full task list at the bottom.

---

## 2.1 Multi-model inference at the edge

Video analytics is increasingly performed on the device that captures the video, such as drones and multi-camera edge boxes, where offloading to the cloud adds prohibitive round-trip latency [[1]](https://arxiv.org/abs/2502.15737). These platforms rely on embedded accelerators such as the NVIDIA Jetson family, with fixed compute and a limited power budget.

Dynamic model selection techniques such as PERTINENCE [[2]](https://doi.org/10.1109/ACCESS.2026.3707342) reduce per-frame computation by routing each input to the cheapest model in a pool expected to handle it correctly, and have been demonstrated on real-time video from intersection cameras for road occupancy estimation [[2]](https://doi.org/10.1109/ACCESS.2026.3707342). As a result, a single video stream becomes several inference workloads, one per model, sharing a single accelerator.

## 2.2 Batching and energy on edge accelerators

Batching amortizes fixed per-invocation costs, so batch execution time grows sublinearly with batch size until the accelerator saturates; EdgeServing reports that increasing batch size from 1 to 10 raises latency by only 2 to 3x on an RTX 3080 [[3]](https://arxiv.org/abs/2605.05527). DNN inference time is also highly predictable [[4]](https://www.usenix.org/conference/osdi20/presentation/gujarati), [[3]](https://arxiv.org/abs/2605.05527), so these curves can be profiled offline. Figure 1 shows per-frame execution time versus batch size for our model pool on a Jetson. `[TODO: Jetson profile, Fig. 1]`

On battery-powered platforms, energy directly limits operating time. Onboard computation affects drone mission time and energy [[5]](https://arxiv.org/abs/1906.10513), and energy is a primary constraint for battery-powered edge cameras [[6]](https://arxiv.org/abs/2506.00754). `[TODO: Jetson busy vs idle power from tegrastats, to justify busy time as marginal energy]`

## 2.3 The batch fragmentation problem

When all frames of a stream are served by one model, they share one queue and batches fill quickly. When the same stream is routed across N models, arrivals are split across N queues, each fills more slowly, and batches shrink. Dynamic model selection thus reduces the work per frame while reducing the efficiency of executing that work.

A scheduler can recover batching efficiency by waiting for a queue to accumulate a larger batch, at the cost of latency. Since the accelerator serves one batch at a time, waiting on one queue also delays every other queue. Navigating this requires deciding, at each scheduling point, which queue to serve, at what batch size, or whether to wait, while accounting for both latency and energy. `[TODO: Fig. 2, latency vs energy tradeoff showing waiting saves energy and costs latency]`

## 2.4 Limitations of existing approaches

**Multi-model serving.** Clockwork [[4]](https://www.usenix.org/conference/osdi20/presentation/gujarati) schedules inference across many models under per-request SLOs, executing the largest deadline-feasible batch among queued requests, and targets datacenter GPU clusters. Symphony [[7]](https://arxiv.org/abs/2308.07470) defers each model's batch until its oldest request approaches the deadline, but schedules each queue independently `[VERIFY: description taken from EdgeServing, read Symphony]`. EdgeServing [[3]](https://arxiv.org/abs/2605.05527) targets single-GPU edge devices and selects the queue minimizing a predicted system-wide urgency score, but always dispatches the maximum available batch, min(|Q_m|, B_max), as soon as the GPU frees, excluding future arrivals from its decision. All three optimize SLO attainment and throughput; none include energy in their objective. EdgeServing additionally locks GPU clocks to their maximum on all platforms, including the Jetson Orin Nano [[3]](https://arxiv.org/abs/2605.05527).

**Energy-aware edge inference.** E4 [[8]](https://ojs.aaai.org/index.php/AAAI/article/view/32104) reduces energy in edge video analytics using early exits and DVFS `[VERIFY]`, PolyThrottle [[9]](https://arxiv.org/abs/2310.19991) tunes CPU, GPU and memory frequencies under application constraints `[VERIFY]`, and Pagoda [[10]](https://arxiv.org/abs/2509.20189) characterizes energy and time across Jetson power modes. These target a single network rather than batches across multiple model queues; their knobs are orthogonal to batch scheduling and could be combined with it.

**Table 1.** Comparison with prior work. `[VERIFY every cell against the full papers]`

| | Multi-model | Edge HW | Energy objective | Chooses batch size | Waits deliberately |
|---|---|---|---|---|---|
| Clockwork [[4]](https://www.usenix.org/conference/osdi20/presentation/gujarati) | ✓ | ✗ | ✗ | ✓ deadline-greedy | ✗ |
| Symphony [[7]](https://arxiv.org/abs/2308.07470) | ✓ | ✗ `[VERIFY]` | ✗ | via deferral | ✓ per queue, deadline-driven |
| EdgeServing [[3]](https://arxiv.org/abs/2605.05527) | ✓ | ✓ | ✗ | ✗ always max | ✗ |
| E4 [[8]](https://ojs.aaai.org/index.php/AAAI/article/view/32104) / PolyThrottle [[9]](https://arxiv.org/abs/2310.19991) | ✗ | ✓ | ✓ | ✗ | ✗ |
| **Ours** | ✓ | ✓ | ✓ | ✓ | ✓ across coupled queues, energy-aware |

## 2.5 Opportunity

To quantify the cost of these limitations, we implement the scheduling policies of Clockwork, Symphony and EdgeServing in a common runtime on a Jetson `[TODO: which model]`, and serve identical arrival traces derived from PERTINENCE routing across a range of load levels and latency budgets, measuring latency and energy using the on-board power monitors. `[TODO: results, Fig. 3]` We expect the largest inefficiency at low and moderate load, where dispatching the maximum available batch immediately produces small batches that could have grown within the latency budget. `[TODO: confirm or revise this expectation with data]`

---

## References

1. L. Rey, A. M. Bernardos, A. D. Dobrzycki, D. Carramiñana, L. Bergesio, J. A. Besada, J. R. Casar. "A Performance Analysis of You Only Look Once Models for Deployment on Constrained Computational Edge Devices in Drone Applications." *Electronics* (MDPI), 2025. arXiv:2502.15737. https://arxiv.org/abs/2502.15737
2. O. Shende, G. Ananthanarayanan, M. Traiola. "PERTINENCE: Input-Based Opportunistic Neural Network Dynamic Execution." *IEEE Access*, 2026. https://doi.org/10.1109/ACCESS.2026.3707342
3. J. Cao, X. Li, Q. Liu, T. Han, N. Zhang, W. Shi. "EdgeServing: Deadline-Aware Multi-DNN Serving at the Edge." arXiv:2605.05527, 2026. https://arxiv.org/abs/2605.05527 `[VERIFY: venue, possibly IEEE ICCCN 2026]`
4. A. Gujarati, R. Karimi, S. Alzayat, W. Hao, A. Kaufmann, Y. Vigfusson, J. Mace. "Serving DNNs like Clockwork: Performance Predictability from the Bottom Up." *OSDI*, 2020. https://www.usenix.org/conference/osdi20/presentation/gujarati
5. B. Boroujerdian, H. Genc, S. Krishnan, B. P. Duisterhof, B. Plancher, K. Mansoorshahi, M. Almeida, W. Cui, A. Faust, V. J. Reddi. "The Role of Compute in Autonomous Aerial Vehicles." arXiv:1906.10513, 2019. https://arxiv.org/abs/1906.10513
6. B. Civjan, B. Chen, R. Zhang, K. Nahrstedt. "EcoLens: Leveraging Multi-Objective Bayesian Optimization for Energy-Efficient Video Processing on Edge Devices." arXiv:2506.00754, 2025. https://arxiv.org/abs/2506.00754
7. L. Chen, W. Deng, A. Canumalla, Y. Xin, D. Zhuo, M. Philipose, A. Krishnamurthy. "Symphony: Optimized DNN Model Serving Using Deferred Batch Scheduling." arXiv:2308.07470, 2024. https://arxiv.org/abs/2308.07470
8. Z. Zhang, Y. Zhao, M.-C. Chang, C. Lin, J. Liu. "E4: Energy-Efficient DNN Inference for Edge Video Analytics via Early Exiting and DVFS." *AAAI*, 2025. https://ojs.aaai.org/index.php/AAAI/article/view/32104
9. M. Yan, H. Wang, S. Venkataraman. "PolyThrottle: Energy-efficient Neural Network Inference on Edge Devices." arXiv:2310.19991, 2023. https://arxiv.org/abs/2310.19991 `[VERIFY: published venue]`
10. Prashanthi S. K., K. K. Sahoo, A. R. Saikia, P. Gupta, A. V. Joshi, P. Pansari, Y. Simmhan. "Pagoda: An Energy and Time Roofline Study for DNN Workloads on Edge Accelerators." arXiv:2509.20189, 2025. https://arxiv.org/abs/2509.20189

---

## Pending tasks

### Reading / verification
- [ ] Read Symphony: confirm deferral mechanism, per-queue independence, hardware (datacenter?), objective
- [ ] Read E4: confirm single-network scope, hardware, no batching / multi-queue
- [ ] Read PolyThrottle: confirm knobs, hardware, no batching; find published venue
- [ ] Read Pagoda beyond abstract: any batch-size results on Jetson?
- [ ] Confirm EdgeServing venue (ICCCN 2026?)
- [ ] Re-check every Table 1 cell against the full papers
- [ ] Pick 1 to 2 more energy use-case citations if 2.2 feels thin (multi-camera / solar camera traps)

### Hardware / data
- [ ] Get Jetson access (Orin Nano or AGX Orin), settle model and power mode
- [ ] Profile T_i(b) for both pools (ResNet primary, YOLO secondary), clocks locked, power mode recorded → Fig. 1
- [ ] Predictability check: p99/p50 per model per batch size on Jetson
- [ ] Measure busy vs idle power via tegrastats → justify energy metric in 2.2
- [ ] Derive arrival mix from PERTINENCE routing splits (ResNet/ImageNette dispatcher, YOLO recall >= 0.80 routing)

### Runtime / experiments
- [ ] Build Jetson runtime with policy/backend separation
- [ ] Implement Clockwork, Symphony, EdgeServing policies (note omissions: no early exit, no LOAD machinery)
- [ ] Fidelity check: EdgeServing policy on its own setup roughly reproduces its published Jetson trend
- [ ] Latency vs energy tradeoff measurement → Fig. 2
- [ ] Load sweep x latency-budget sweep, all baselines → Fig. 3, fill 2.5
- [ ] Sweep baselines' static knobs (SLO, defer threshold) to pre-empt "just tune the baseline"
- [ ] Optional: offline oracle bound from the profile table over recorded traces

### Writing
- [ ] Differentiate our wait-cost from EdgeServing's stability score f(w) = min(exp(w/tau - 1), C) (belongs in Design, but must exist)
- [ ] Confirm ResNet-primary / YOLO-secondary split with profs at meet 6
- [ ] Decide whether to keep any YOLOv8l-only reference line (with accuracy reported alongside)
