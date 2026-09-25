# Week 5

## Meeting notes & tasks

**Tasks**:
1. Literature search: find whether the same or a similar scheduling problem has already been
   solved. If the same work exists, adapt it to our setting rather than rebuild it.
2. Build a discrete event simulator for the scheduler. The scheduling problem is decoupled from
   PERTINENCE itself, so it gets designed and validated in simulation first.
3. Once the scheduler works in simulation, prove it experimentally with PERTINENCE in the loop.
4. Mail the meeting notes to Prof. Gayathri and ask for a group chat for occasional quick
   opinions between meetings.

Target setting: scheduling on compute-constrained edge devices (Jetsons), not large-scale
datacenter serving.

Later extensions (not this cycle):
- Real-time constraints: per-job deadlines.
- Priority-based operation: some video streams matter more than others.

Next meeting (meet 6) is in two weeks, no meeting in between.

---

## [MEETING] Scheduler direction accepted, simulator first

Presented the formal problem definition and the YOLO correctness work (Week 4). The direction was
received well.

The literature review was presented as still to do. No prior-art findings were shown in this
meeting.

**Key decision**: the scheduling problem stands on its own. The scheduler only sees queues, batch
sizes and runtimes T_i(b), not how frames got routed. So it gets built and evaluated in a discrete
event simulator first, and only then validated experimentally with PERTINENCE doing the routing.

**Scope**: optimise for edge devices (Jetsons). This separates the work from most of the existing
scheduling and serving literature, which targets large-scale, datacenter-sized optimisation. The
constraint here is limited compute on one device.

**Extensions** raised for later: deadlines (real-time constraints) and stream priorities.

## Log

## [DECISION] Compute metric: report both compression ratio and busy time, pick the primary experimentally

The Week 4 journal defined compute as accelerator busy time, while the Week 4 deck used
compression ratio. Settling it before the simulator, since it's the number every policy gets
scored on.

**The two candidates**:
- *Compression ratio* K / I: batches run per job processed (1 / mean batch size). K / I = 1
  means no batching.
- *Busy time* C = sum of T_i(b) over every batch run: total time the accelerator spends computing.

**When they agree**: if a model's runtime is affine, T(b) = α + βb, total busy time is
α·K + β·I. I is fixed, so for a single model, minimizing busy time and minimizing K are the same
thing.

**When they don't**:
- Convex curves: once per-image cost rises with batch size, K rewards batches that are slower. On
  the Week 2 resnet50 numbers, one batch of 32 takes 149.6 ms while four batches of 8 take
  109.0 ms, yet K prefers the batch of 32.
- A heterogeneous pool: K counts a nano batch and a large batch as one unit each, so a lower K
  across models doesn't mean less compute.

**Decision**: the simulator reports both. Only one will be the primary objective, and which one
gets decided experimentally: run the candidate policies on the measured YOLOv8 T_i(b) curves and
check whether ranking policies by K / I and by busy time gives the same order. If they agree over
the batch sizes we allow, compression ratio is the simpler, hardware-independent choice. If they
diverge, the divergence itself is the result to look at.
