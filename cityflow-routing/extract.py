"""
Lists the CityFlowV2 zip and extracts only the chosen scenarios: each camera's
vdo.avi plus the small timing files (cam_timestamp, cam_framenum) and the
dataset's ReadMe.txt / list_cam.txt. Everything else (gt, det, mtsc, segm,
roi) stays in the zip.

Run:
    python extract.py --list                 # scenario table only
    python extract.py --scenarios S01        # default: S01
    python extract.py --scenarios S01 S02 --delete-zip-after
"""

import argparse
import os
import shutil
import sys
import zipfile

import dataset


def list_scenarios(zf):
    """{scenario: {"split", "cameras": [names], "bytes", "videos": [zip entries]}}"""
    scenarios = {}
    for info in zf.infolist():
        match = dataset.VIDEO_PATTERN.match(info.filename)
        if not match:
            continue
        split, scenario, camera = match.groups()
        entry = scenarios.setdefault(scenario, {"split": split, "cameras": [], "bytes": 0, "videos": []})
        entry["cameras"].append(camera)
        entry["bytes"] += info.file_size
        entry["videos"].append(info)
    return dict(sorted(scenarios.items()))


def print_table(scenarios):
    print(f"{'scenario':<9} {'split':<11} {'cameras':>7} {'video GB':>9}")
    for name, s in scenarios.items():
        print(f"{name:<9} {s['split']:<11} {len(s['cameras']):>7} {s['bytes'] / 1e9:>9.2f}")


def extract_member(zf, info, out_root):
    """Extracts one zip entry to out_root/<its path>, skipping it if a file of
    the right size is already there. Writes to a .part file, then renames."""
    dest = os.path.join(out_root, *info.filename.split("/"))
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with zf.open(info) as src, open(dest + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst, length=1 << 20)
    os.replace(dest + ".part", dest)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=dataset.DEFAULT_DATA_DIR)
    parser.add_argument("--scenarios", nargs="+", default=["S01"])
    parser.add_argument("--list", action="store_true", help="only print the scenario table")
    parser.add_argument("--delete-zip-after", action="store_true")
    args = parser.parse_args(argv)

    path = dataset.zip_path(args.data_dir)
    if not os.path.exists(path):
        sys.exit(f"{path} not found: run download.py first")

    with zipfile.ZipFile(path) as zf:
        scenarios = list_scenarios(zf)
        print_table(scenarios)
        if args.list:
            return

        unknown = [s for s in args.scenarios if s not in scenarios]
        if unknown:
            sys.exit(f"Not in the zip: {unknown}")

        names = set(zf.namelist())
        members = []
        for scenario in args.scenarios:
            members += scenarios[scenario]["videos"]
            for small in [f"cam_timestamp/{scenario}.txt", f"cam_framenum/{scenario}.txt"]:
                if small not in names:
                    sys.exit(f"{small} missing from the zip")
                members.append(zf.getinfo(small))
        for extra in ["ReadMe.txt", "list_cam.txt"]:
            if extra in names:
                members.append(zf.getinfo(extra))

        out_root = dataset.extracted_dir(args.data_dir)
        os.makedirs(out_root, exist_ok=True)
        total_gb = sum(m.file_size for m in members) / 1e9
        free_gb = shutil.disk_usage(out_root).free / 1e9
        print(f"\nExtracting {len(members)} files ({total_gb:.2f} GB) for {args.scenarios} "
              f"into {out_root} ({free_gb:.1f} GB free)")
        if free_gb < total_gb:
            sys.exit("Not enough free disk.")

        for i, info in enumerate(members, 1):
            written = extract_member(zf, info, out_root)
            status = "extracted" if written else "already there"
            print(f"  [{i}/{len(members)}] {info.filename} ({info.file_size / 1e6:.0f} MB) {status}")

    if args.delete_zip_after:
        os.remove(path)
        print(f"Deleted {path} (other scenarios now need a fresh download)")
    print("Done.")


if __name__ == "__main__":
    main()
