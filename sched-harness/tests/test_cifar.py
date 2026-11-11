# Tests for the CIFAR input pool (common/cifar.py) and make_cifar128.py. Plain asserts, no torch, no board.
# Run: python3 tests/test_cifar.py
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from common import cifar


def test_resize_shape_and_dtype():
    out = cifar.resize_bilinear(np.zeros((32, 32, 3), np.uint8), 224)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


def test_resize_keeps_a_constant_image_constant():
    out = cifar.resize_bilinear(np.full((32, 32, 3), 77, np.uint8), 224)
    assert (out == 77).all()


def test_resize_ramp_is_monotone_and_keeps_the_ends():
    ramp = np.tile(np.linspace(0, 255, 32).astype(np.uint8)[None, :, None], (32, 1, 3))
    out = cifar.resize_bilinear(ramp, 224)
    row = out[100, :, 0].astype(int)
    assert (np.diff(row) >= 0).all()
    assert row[0] == 0 and row[-1] == 255


def _fake_npz(path, n=130):
    rs = np.random.RandomState(1)
    images = rs.randint(0, 256, (n, 32, 32, 3), dtype=np.uint8)
    images[0] = 0
    images[0, :, :, 0], images[0, :, :, 1], images[0, :, :, 2] = 10, 20, 30      # R, G, B constants
    np.savez(path, images=images, labels=np.zeros(n, np.int64), indices=np.arange(n), seed=np.int64(0))


def test_load_pool_is_bgr_224_and_takes_the_first_count():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.npz")
        _fake_npz(p)
        pool, text = cifar.load_pool(p, 224, 128)
        assert len(pool) == 128 and pool[0].shape == (224, 224, 3) and pool[0].flags["C_CONTIGUOUS"]
        assert tuple(pool[0][5, 5]) == (30, 20, 10)           # stored BGR, BatchBuffer flips it to RGB
        assert "CIFAR-10" in text and "128" in text


def test_load_pool_errors_are_readable():
    with tempfile.TemporaryDirectory() as d:
        try:
            cifar.load_pool(os.path.join(d, "none.npz"), 224, 128)
            assert False
        except FileNotFoundError as e:
            assert "make_cifar128.py" in str(e)
        p = os.path.join(d, "x.npz")
        _fake_npz(p, n=64)
        try:
            cifar.load_pool(p, 224, 128)
            assert False
        except ValueError as e:
            assert "64" in str(e)


def test_make_script_reads_a_cifar_shaped_tarball():
    with tempfile.TemporaryDirectory() as d:
        rs = np.random.RandomState(2)
        data = rs.randint(0, 256, (10000, 3072), dtype=np.uint8)
        os.makedirs(os.path.join(d, "cifar-10-batches-py"))
        with open(os.path.join(d, "cifar-10-batches-py", "test_batch"), "wb") as fh:
            pickle.dump({b"data": data, b"labels": [int(x) for x in rs.randint(0, 10, 10000)]}, fh)
        tar_path = os.path.join(d, "c.tar.gz")
        with tarfile.open(tar_path, "w:gz") as t:
            t.add(os.path.join(d, "cifar-10-batches-py"), arcname="cifar-10-batches-py")
        out = os.path.join(d, "o.npz")
        subprocess.check_call([sys.executable, os.path.join(ROOT, "make_cifar128.py"), "--tar", tar_path, "--out", out],
                              stdout=subprocess.DEVNULL)
        with np.load(out) as z:
            assert z["images"].shape == (128, 32, 32, 3) and len(set(z["indices"].tolist())) == 128
            i = int(z["indices"][3])
            want = data[i].reshape(3, 32, 32).transpose(1, 2, 0)      # plane layout to HWC
            assert (z["images"][3] == want).all()


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    failed = 0
    for n in names:
        try:
            globals()[n]()
            print("PASS", n)
        except AssertionError as e:
            failed += 1
            print("FAIL", n, e)
    print("%d tests, %d failed" % (len(names), failed))
    sys.exit(1 if failed else 0)
