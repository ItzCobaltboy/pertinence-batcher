# The CIFAR-10 input pool: 128 random test images kept in a small .npz, upscaled once to the model input size.
# Torch free on purpose, so it can be tested anywhere. make_cifar128.py builds the .npz.
import os

import numpy as np


def resize_bilinear(img, size):
    """(h, w, 3) uint8 -> (size, size, 3) uint8. Bilinear with pixel centres aligned (torch align_corners=False)."""
    h, w = img.shape[:2]

    def axis(n_in, n_out):
        pos = np.clip((np.arange(n_out) + 0.5) * n_in / float(n_out) - 0.5, 0, n_in - 1)
        lo = np.floor(pos).astype(int)
        return lo, np.minimum(lo + 1, n_in - 1), (pos - lo).astype(np.float32)

    y0, y1, wy = axis(h, size)
    x0, x1, wx = axis(w, size)
    f = img.astype(np.float32)
    top = f[y0][:, x0] * (1 - wx)[None, :, None] + f[y0][:, x1] * wx[None, :, None]
    bot = f[y1][:, x0] * (1 - wx)[None, :, None] + f[y1][:, x1] * wx[None, :, None]
    out = top * (1 - wy)[:, None, None] + bot * wy[:, None, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def load_pool(path, size, count):
    """First `count` images of the .npz, resized to size x size and stored BGR (BatchBuffer flips BGR to RGB).
    Returns (list of contiguous uint8 arrays, source text)."""
    if not os.path.exists(path):
        raise FileNotFoundError("%s is missing. Build it once where the network is open: python3 make_cifar128.py "
                                "(or --tar cifar-10-python.tar.gz), then copy data/cifar10_128.npz here." % path)
    with np.load(path) as z:
        images, indices, seed = z["images"], z["indices"], int(z["seed"])
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError("%s: expected uint8 (N, 32, 32, 3), got %s %s" % (path, images.dtype, images.shape))
    if len(images) < count:
        raise ValueError("%s has %d images, INPUT_POOL_SIZE wants %d" % (path, len(images), count))
    pool = [np.ascontiguousarray(resize_bilinear(im, size)[:, :, ::-1]) for im in images[:count]]
    text = "CIFAR-10 test set, %d random images (seed %d, first index %d), 32x32 -> %dx%d bilinear" % (
        count, seed, int(indices[0]), size, size)
    return pool, text
