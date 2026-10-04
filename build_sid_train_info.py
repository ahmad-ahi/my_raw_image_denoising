"""
Build a training info file from SID's official Sony_train_list.txt.

This extracts the UNIQUE clean (long-exposure) images referenced by the
official SID training split only -- val/test images are never touched,
so there is no leakage into SID_evaltest.info or any eval benchmark.

Output format matches what SimpleTrainDataset (in train.py) already expects:
a list of "scenes", where each scene is a list of dicts with at least
{"data": <path>, "ISO": <int>, "ratio": 1}. Since we only care about clean
frames here, each scene contains exactly one item with ratio=1.

Usage:
    python3 build_sid_train_info.py \
        --sid_root /home/am56/data/SID \
        --train_list Sony_train_list.txt \
        --out infos/SID_train.info
"""
import argparse
import os
import pickle
import re


def parse_train_list(list_path, sid_root):
    """Parse Sony_train_list.txt and return unique (abs_path, iso) pairs."""
    seen = {}
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # parts[1] is the long (clean) path, parts[2] is like "ISO200"
            long_rel = parts[1].lstrip("./")
            iso_str = parts[2]
            m = re.match(r"ISO(\d+)", iso_str)
            if not m:
                raise ValueError(f"Could not parse ISO from '{iso_str}' in line: {line}")
            iso = int(m.group(1))

            abs_path = os.path.join(sid_root, long_rel)
            # Same long image may appear multiple times (paired with
            # different short exposures) -- dedupe by path.
            seen[abs_path] = iso
    return seen


def main(args):
    pairs = parse_train_list(args.train_list, args.sid_root)
    print(f"Found {len(pairs)} unique clean training images.")

    missing = [p for p in pairs if not os.path.exists(p)]
    if missing:
        print(f"WARNING: {len(missing)} referenced files do not exist on disk, e.g.:")
        for p in missing[:5]:
            print(f"  {p}")
        print("These will be skipped.")

    scenes = []
    n_skipped = 0
    for path, iso in pairs.items():
        if path in missing:
            n_skipped += 1
            continue
        scenes.append([{
            "data": path,
            "ISO": iso,
            "ratio": 1,
        }])

    with open(args.out, "wb") as f:
        pickle.dump(scenes, f)

    print(f"Wrote {len(scenes)} scenes to {args.out} "
          f"({n_skipped} skipped due to missing files).")


def rebuild_from_flat_dir(flat_dir, out_path):
    """Rebuild the .info file to point at a flat directory of already-copied
    long files (e.g. after copying just the 161 needed files out of the full
    68GB SID dataset for a smaller, more portable subset). ISO is re-parsed
    from each existing .info entry by matching filenames, since the flat
    directory itself doesn't carry ISO metadata."""
    with open(out_path, "rb") as f:
        old_data = pickle.load(f)

    iso_by_basename = {}
    for scene in old_data:
        for item in scene:
            iso_by_basename[os.path.basename(item["data"])] = item["ISO"]

    scenes = []
    n_missing = 0
    for fname, iso in iso_by_basename.items():
        new_path = os.path.join(flat_dir, fname)
        if not os.path.exists(new_path):
            n_missing += 1
            continue
        scenes.append([{"data": new_path, "ISO": iso, "ratio": 1}])

    with open(out_path, "wb") as f:
        pickle.dump(scenes, f)

    print(f"Rebuilt {out_path}: {len(scenes)} scenes pointing at '{flat_dir}' "
          f"({n_missing} missing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid_root", type=str, default=None,
                         help="Path to the SID dataset root, e.g. /home/am56/data/SID "
                              "(the directory containing Sony/long, Sony/short, etc.). "
                              "Required unless --rebuild_flat_dir is given.")
    parser.add_argument("--train_list", type=str, default=None,
                         help="Path to Sony_train_list.txt. Defaults to "
                              "<sid_root>/Sony_train_list.txt if not given.")
    parser.add_argument("--out", type=str, default="infos/SID_train.info")
    parser.add_argument("--rebuild_flat_dir", type=str, default=None,
                         help="If given, skip parsing Sony_train_list.txt entirely and "
                              "instead rewrite an EXISTING --out file (read then "
                              "overwritten) to point at this flat directory of "
                              "already-copied long files, matched by filename.")
    args = parser.parse_args()

    if args.rebuild_flat_dir:
        rebuild_from_flat_dir(args.rebuild_flat_dir, args.out)
    else:
        if not args.sid_root:
            parser.error("--sid_root is required unless --rebuild_flat_dir is given")
        if args.train_list is None:
            args.train_list = os.path.join(args.sid_root, "Sony_train_list.txt")
        main(args)