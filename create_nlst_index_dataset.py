import argparse, csv, os, re, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter, defaultdict
import csv

# -------------------------------------------------------------------------
# 1.  Filter-code catalogue  (regex pattern ➜ NLST code)
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# 1-12 reconstruction-filter codes (“most-specific” → “fallback”)
# -------------------------------------------------------------------------
FILTER_MAP = [
    # Siemens sharp kernels
    (re.compile(r"\bb50f?\b",     re.I), "7"),   # B50/B50f
    # Siemens soft kernels (B3x family)
    (re.compile(r"\bb3[0-9]f?\b", re.I), "8"),   # B30-B39, B31f …

    # Siemens – other
    (re.compile(r"(spr|lspr)",    re.I), "9"),   # LSPR16, SPR, …
    (re.compile(r"siem",          re.I), "9"),

    # GE
    (re.compile(r"bone",          re.I), "1"),   # BONE, BONE3, BONEPLUS …
    (re.compile(r"stand",         re.I), "2"),   # STANDARD, STD, STAND30 …
    (re.compile(r"\bge\b",        re.I), "3"),   # GE, other   (keep word bounds!)

    # Philips kernels
    (re.compile(r"phil.*d",       re.I), "4"),   # Philips D  (Br64D, PhilDLu …)
    (re.compile(r"phil.*c",       re.I), "5"),   # Philips C
    # NEW: Philips MX series (MX8000D / MX8000C)
    (re.compile(r"mx[0-9]*.*d",   re.I), "4"),   # MX8000D …
    (re.compile(r"mx[0-9]*.*c",   re.I), "5"),   # MX8000C …

    (re.compile(r"phil",          re.I), "6"),   # Philips, other

    # Toshiba
    (re.compile(r"fc10",          re.I), "10"),  # FC10
    (re.compile(r"fc51",          re.I), "11"),  # FC51
    (re.compile(r"tosh",          re.I), "12"),  # Toshiba, other
]

MISSING_CODE = "M"

LOCALIZER_PAT = re.compile(r"(local|scout)", re.I)

# -------------------------------------------------------------------------
def extract_filter(description: str) -> str:
    """Return NLST filter code (“1”…“12”, “M” if unknown)."""
    for pat, code in FILTER_MAP:
        if pat.search(description):
            return code
    return MISSING_CODE


def is_localizer(series_path: Path) -> bool:
    """
    A folder is a localizer if:
    * name contains 'local' or 'scout' **or**
    * it has < 10 DICOM slices.
    """
    if LOCALIZER_PAT.search(series_path.name):
        return True

    # cheap slice count – break at 10
    n = 0
    with os.scandir(series_path) as it:
        for entry in it:
            if entry.name.lower().endswith(".dcm"):
                n += 1
                if n >= 10:
                    return False   # not a localizer
    return True                   # we saw < 10 .dcm files


def sort_timepoints(tp_dirs: List[Path]) -> List[Path]:
    """
    Return *tp_dirs* sorted chronologically, trying to parse a leading
    date like '01-02-1999' or '1999-02-01'.  If parsing fails, fallback
    to lexical sort.
    """
    def tp_key(p: Path):
        # grab first 10-char block that looks like a date
        token = p.name[:10]
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                continue
        return p.name  # lexical fallback

    return sorted(tp_dirs, key=tp_key)


def map_timepoints(tp_dirs_sorted: List[Path]) -> Dict[str, Path]:
    """
    Map first three time-points to t0/t1/t2 Path objects.
    Keys missing if <3 time-points.
    """
    mapping = {}
    for i, p in enumerate(tp_dirs_sorted[:3]):
        mapping[f"t{i}"] = p
    return mapping


# -------------------------------------------------------------------------
def make_index(nlst_root: Path) -> List[Dict[str, str]]:
    rows = []

    # ─── iterate over PIDs ────────────────────────────────────────────────
    for pid_dir in sorted(nlst_root.iterdir()):
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name

        # discover time-point folders (level-2 subdirs)
        tp_dirs = [d for d in pid_dir.iterdir() if d.is_dir()]
        if not tp_dirs:
            continue

        tp_dirs_sorted = sort_timepoints(tp_dirs)
        tp_map = map_timepoints(tp_dirs_sorted)           # 't0' → Path, …

        # ─── for each time-point, enumerate series ───────────────────────
        for tp_key, tp_path in tp_map.items():
            for series_dir in tp_path.iterdir():
                if not series_dir.is_dir():
                    continue
                if is_localizer(series_dir):
                    continue

                # SeriesDescription is the bit after first '-' and before second '-'
                #   <num>.000000-<SeriesDescription>-<UID>
                parts = series_dir.name.split("-", 2)
                description = parts[1] if len(parts) >= 2 else series_dir.name
                filt_code = extract_filter(description)

                row = {
                    "pid": pid,
                    "t0":  str(series_dir) if tp_key == "t0" else "",
                    "t1":  str(series_dir) if tp_key == "t1" else "",
                    "t2":  str(series_dir) if tp_key == "t2" else "",
                    "filter": filt_code,
                }
                rows.append(row)

    return rows


# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Root NLST folder (…/NLST)")
    ap.add_argument("csv_out",   help="Output CSV file")
    args = ap.parse_args()

    nlst_root = Path(args.input_dir).expanduser().resolve()
    if not nlst_root.is_dir():
        sys.exit(f"Input directory {nlst_root} not found.")

    rows = make_index(nlst_root)

    # ─── write CSV ────────────────────────────────────────────────────────
    fieldnames = ["pid", "t0", "t1", "t2", "filter"]
    with open(args.csv_out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
    counts = Counter()
    with open("nlst_index.csv") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            counts[row["filter"]] += 1
    print(counts)
