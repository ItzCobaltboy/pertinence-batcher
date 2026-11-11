#!/usr/bin/env python3
"""Builds data/cifar10_128.npz: 128 random images of the CIFAR-10 test set (numpy only, no torch needed).
Run it once on any machine with network, then copy data/cifar10_128.npz to the board.
  python3 make_cifar128.py                       downloads cifar-10-python.tar.gz (163 MB) from the official page
  python3 make_cifar128.py --tar path.tar.gz     uses a tarball you already have
  python3 make_cifar128.py --batch-file test_batch   uses the test_batch file from cifar-10-batches-py"""
import argparse
import hashlib
import io
import os
import pickle
import sys
import tarfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def read_test_batch(args):
    if args.batch_file:
        with open(args.batch_file, "rb") as fh:
            return pickle.load(fh, encoding="bytes")
    tar_path = args.tar
    if not tar_path:
        tar_path = os.path.join(HERE, "data", "cifar-10-python.tar.gz")
        os.makedirs(os.path.dirname(tar_path), exist_ok=True)
        print("downloading %s" % URL)
        try:
            urllib.request.urlretrieve(URL, tar_path)
        except Exception as e:
            sys.exit("download failed (%s). Get the tarball another way and pass --tar, or pass --batch-file." % e)
    with tarfile.open(tar_path) as tar:
        member = [m for m in tar.getmembers() if m.name.endswith("test_batch")]
        if len(member) != 1:
            sys.exit("no single test_batch in %s" % tar_path)
        return pickle.load(io.BytesIO(tar.extractfile(member[0]).read()), encoding="bytes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar")
    ap.add_argument("--batch-file")
    ap.add_argument("--count", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "data", "cifar10_128.npz"))
    args = ap.parse_args()

    batch = read_test_batch(args)
    data = np.asarray(batch[b"data"], dtype=np.uint8)            # (10000, 3072): R plane, G plane, B plane
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    if data.shape != (10000, 3072):
        sys.exit("unexpected test_batch shape %s" % (data.shape,))
    images = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)   # NHWC, RGB
    indices = np.random.RandomState(args.seed).choice(len(images), args.count, replace=False)
    out_images = np.ascontiguousarray(images[indices])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, images=out_images, labels=labels[indices], indices=indices, seed=np.int64(args.seed))
    digest = hashlib.sha1(out_images.tobytes()).hexdigest()
    print("wrote %s: %d images, seed %d, sha1 of pixels %s" % (args.out, args.count, args.seed, digest))
    print("images per class:", np.bincount(labels[indices], minlength=10).tolist())


if __name__ == "__main__":
    main()
