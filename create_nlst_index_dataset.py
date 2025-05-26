import argparse, csv, os, re, sys
from pathlib import Path
from typing import List


FILTER_MAP = [
    (re.compile(r"b50",  re.I), "7"),   # Siemens B50F
    (re.compile(r"b30",  re.I), "8"),   # Siemens B30
    (re.compile(r"siem", re.I), "9"),   # Siemens, other
    (re.compile(r"\bbone\b", re.I), "1"),  # GE Bone
    (re.compile(r"\bstandard\b", re.I), "2"),  # GE Standard
    (re.compile(r"\bge\b",  re.I), "3"),   # GE, other
    (re.compile(r"phill.*d", re.I), "4"),  # Phillips D
    (re.compile(r"phill.*c", re.I), "5"),  # Phillips C
    (re.compile(r"phill",    re.I), "6"),  # Phillips, other
    (re.compile(r"fc10", re.I),  "10"),    # Toshiba FC10
    (re.compile(r"fc51", re.I),  "11"),    # Toshiba FC51
    (re.compile(r"tosh", re.I),  "12"),    # Toshiba, other
]
MISSING_CODE = "M"                         # not matched / <4 filters

LOCALIZER_PAT = re.compile(r"(local|scout)", re.I)


def extract_filter(description: str) -> str:
    """Return NLST filter code string (“1”…“12”, “M”)."""
    for pat, code in FILTER_MAP:
        if pat.search(description):
            return code
    return MISSING_CODE


def series_to_timepoint(series_dirs: List[str]) -> dict:
    """
    Map up to three series names (sorted by numeric prefix) to {'t0':name, ...}.
    Returns a dict with keys 't0', 't1', 't2'.  Missing keys are omitted.
    """
    mapping = {}
    for i, s in enumerate(series_dirs[:3]):   # only the first three
        mapping[f"t{i}"] = s
    return mapping


def is_localizer(series_path: Path) -> bool:
    """
    Return True if the folder looks like a scout/localizer scan.

    Criteria
    --------
    1.  Folder name contains 'local', 'scout', etc.  (regex match)
    2.  The directory holds < 10 .dcm slices.
    """
    name_flag = bool(LOCALIZER_PAT.search(series_path.name))

    # quick slice count – stop as soon as we find 10
    n_slices = 0
    with os.scandir(series_path) as it:
        for entry in it:
            if entry.name.lower().endswith(".dcm"):
                n_slices += 1
                if n_slices >= 10:
                    break
    slice_flag = n_slices < 10

    return name_flag or slice_flag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", help="Root of raw NLST tree (…/NLST)")
    p.add_argument("csv_out",   help="Output CSV path")
    args = p.parse_args()

    in_root = Path(args.input_dir).expanduser().resolve()
    if not in_root.is_dir():
        sys.exit(f"Input directory {in_root} not found.")

    rows = []

    for pid_dir in sorted(in_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name

        # list series, drop localizers, sort by numeric prefix
        series_all = [
            d.name for d in pid_dir.iterdir()
            if d.is_dir() and not is_localizer(d)  # ← pass Path, not name
        ]
        if not series_all:
            continue

        series_all.sort(
            key=lambda s: int(s.split(".", 1)[0])  # numeric prefix
            if s.split(".", 1)[0].isdigit() else 9999
        )

        # map first three to time-points
        tp_map = series_to_timepoint(series_all)

        # build one CSV row per series
        for s in series_all:
            description = s.split("-", 2)[1] if "-" in s else s
            filt_code   = extract_filter(description)

            row = {
                "pid": pid,
                "t0":  tp_map.get("t0") if tp_map.get("t0") == s else "",
                "t1":  tp_map.get("t1") if tp_map.get("t1") == s else "",
                "t2":  tp_map.get("t2") if tp_map.get("t2") == s else "",
                "filter": filt_code,
            }
            rows.append(row)

    fieldnames = ["pid", "t0", "t1", "t2", "filter"]
    with open(args.csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
