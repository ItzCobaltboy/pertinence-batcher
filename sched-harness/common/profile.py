# Reads profile.json (calibrated service times T_i(b)) and answers "how long does a batch of b take on model i".
# No torch here, so the policies, the traffic maths and the checks can use it on the laptop.
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def service_ms(profile, letter, b, stat):
    """Service time of the whole pipeline (h2d + forward + NMS) for a batch of b on model `letter`.
    Linear interpolation between profiled batch sizes; never extrapolates."""
    table = profile["models"][letter]["b"]
    sizes = sorted(int(k) for k in table)
    key = stat + "_ms"
    if b in sizes:
        return table[str(b)]["total"][key]
    if b < sizes[0] or b > sizes[-1]:
        raise ValueError("batch size %d is outside the profiled range %d..%d for model %s" % (b, sizes[0], sizes[-1], letter))
    lo = max(s for s in sizes if s < b)
    hi = min(s for s in sizes if s > b)
    t_lo = table[str(lo)]["total"][key]
    t_hi = table[str(hi)]["total"][key]
    return t_lo + (t_hi - t_lo) * (b - lo) / float(hi - lo)
