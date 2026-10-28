"""Numpy re-implementation of the dispatcher FC training step (same maths as
torch: nn.Linear default init, Adam b=(0.9,0.999) eps=1e-8, mean batch loss of
CE * P[true,pred] * w[true] * 1[pred!=true], shuffled mini-batches)."""
import numpy as np

def train_fc_numpy(X, y, P, w, in_dim, n_cls, epochs, batch, lr, seed):
    rng = np.random.RandomState(seed)
    k = 1.0 / np.sqrt(in_dim)
    W = rng.uniform(-k, k, (n_cls, in_dim)).astype(np.float32)
    b = rng.uniform(-k, k, (n_cls,)).astype(np.float32)
    mW = np.zeros_like(W); vW = np.zeros_like(W); mb = np.zeros_like(b); vb = np.zeros_like(b)
    X = X.astype(np.float32); y = y.astype(np.int64)
    P = np.asarray(P, np.float32); w = np.asarray(w, np.float32)
    t = 0; n = len(X); b1, b2, eps = 0.9, 0.999, 1e-8
    for _ in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]; xb = X[idx]; yb = y[idx]; B = len(idx)
            z = xb @ W.T + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z); p /= p.sum(1, keepdims=True)
            pred = p.argmax(1)
            cw = P[yb, pred] * w[yb] * (pred != yb)
            g = p.copy(); g[np.arange(B), yb] -= 1.0
            g *= (cw / B)[:, None]
            gW = g.T @ xb; gb = g.sum(0)
            t += 1
            for prm, gr, m, v in ((W, gW, mW, vW), (b, gb, mb, vb)):
                m *= b1; m += (1 - b1) * gr
                v *= b2; v += (1 - b2) * gr * gr
                prm -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
    return W, b
