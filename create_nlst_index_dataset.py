from collections import Counter, defaultdict
import argparse, csv, os, re, sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pydicom
from tqdm import tqdm

# ----------------------------------------------------------------------
# NLST kernel-code lookup
# ----------------------------------------------------------------------
FILTER_MAP = [
    # Siemens
    (re.compile(r"b50f?",        re.I), "7"),
    (re.compile(r"b2\d+f?",      re.I), "8"),  # B20–B29
    (re.compile(r"b3\d+f?",      re.I), "8"),  # B30–B39
    (re.compile(r"b7\d+f?",      re.I), "9"),  # B70–B79
    (re.compile(r"(spr|lspr)",   re.I), "9"),
    (re.compile(r"siem",         re.I), "9"),

    # GE
    (re.compile(r"bone",         re.I), "1"),
    (re.compile(r"stand",        re.I), "2"),
    (re.compile(r"lsplus",       re.I), "3"),
    (re.compile(r"lung",         re.I), "3"),
    (re.compile(r"qxd",          re.I), "3"),
    (re.compile(r"\bge\b",       re.I), "3"),

    # Philips
    (re.compile(r"phil.*d",      re.I), "4"),
    (re.compile(r"phil.*c",      re.I), "5"),
    (re.compile(r"phmx.*d",      re.I), "4"),
    (re.compile(r"phmx.*c",      re.I), "5"),
    (re.compile(r"phmx.*b",      re.I), "6"),
    (re.compile(r"phil",         re.I), "6"),

    # Toshiba
    (re.compile(r"fc10",         re.I), "10"),
    (re.compile(r"fc51",         re.I), "11"),
    (re.compile(r"tosh",         re.I), "12"),
]
MISSING_CODE = "M"


def map_kernel(text: str) -> str:
    for pat, code in FILTER_MAP:
        if pat.search(text):
            return code
    return MISSING_CODE


# ────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ────────────────────────────────────────────────────────────────────────
def first_dicom(vol_dir):
    """Return first file that *looks* like DICOM inside *vol_dir*."""
    for f in vol_dir.iterdir():
        if f.is_file() and (f.suffix.lower() == ".dcm" or f.suffix == ""):
            return f
    raise FileNotFoundError(f"No DICOM slices in {vol_dir}")


def series_meta(vol_dir):
    """Return (StudyDate, ConvolutionKernel) from the first slice."""
    ds = pydicom.dcmread(first_dicom(vol_dir),
                         stop_before_pixels=True, force=True)

    date = (ds.get("StudyDate") or
            ds.get("SeriesDate") or
            ds.get("AcquisitionDate") or
            "").strip()
    kernel = ds.get("ConvolutionKernel", "").strip()
    return date, kernel


def sort_timepoints(tp_dirs: List[Path]) -> List[Path]:
    """Sort time-point folders chronologically using a leading date token."""
    def tp_key(p: Path):
        token = p.name[:10]
        for fmt in ("%d-%m-%Y", "%m-%d-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(token, fmt)
            except ValueError:
                continue
        return p.name            # lexical fallback

    return sorted(tp_dirs, key=tp_key)


def assign_tp(items: List[str]) -> Dict[str, str]:
    """Map first 3 unique items to {'t0':item0, 't1':item1, 't2':item2}."""
    mapping = {}
    for i, v in enumerate(items[:3]):
        mapping[f"t{i}"] = v
    return mapping


def get_npy_path(volume_path, img_root="/local/amvepa91/nlst_npy"):
    volume_name = os.path.basename(volume_path)
    time_point_dir = os.path.basename(os.path.dirname(volume_path))
    pid_dir = os.path.basename(os.path.dirname(os.path.dirname(volume_path)))
    volume_path_npy = os.path.join(img_root, pid_dir, time_point_dir, volume_name + ".npy")
    return volume_path_npy


# ────────────────────────────────────────────────────────────────────────
# 3.  Per-PID processing
# ────────────────────────────────────────────────────────────────────────
def rows_for_pid(pid_dir: Path, min_slices=20) -> List[Dict[str, str]]:
    pid = pid_dir.name
    tp_dirs = sort_timepoints([d for d in pid_dir.iterdir() if d.is_dir()])
    if not tp_dirs:
        return []

    orig_tp_map = assign_tp([d.name for d in tp_dirs])        # by folder order

    # collect metadata for every volume (=series)
    series_info = []  # List[(vol_path, tp_name, study_date, kernel)]
    for tp in tp_dirs:
        for vol in tp.iterdir():
            if not vol.is_dir():
                continue
            volume_path = vol.resolve()
            volume_path_npy = get_npy_path(volume_path)
            if not os.path.exists(volume_path_npy):
                print(f"⚠️  Skip {vol} (no .npy found)", file=sys.stderr)
                continue

            # ---------- localizer filter (skip small volumes) ----------
            n_slices = 0
            for f in vol.iterdir():
                if f.is_file() and (f.suffix.lower() == ".dcm" or f.suffix == ""):
                    n_slices += 1
                    if n_slices >= min_slices:                 # threshold here
                        break
            if n_slices < min_slices:
                continue
            # -----------------------------------------------------------

            try:
                date, kernel = series_meta(vol)
            except Exception as e:
                print(f"⚠️  Skip {vol}  ({e})", file=sys.stderr)
                continue
            series_info.append((vol, tp.name, date, kernel))

    if not series_info:
        return []

    # derive DICOM-based time-points (unique StudyDates)
    unique_dates = sorted({d for _, _, d, _ in series_info if d})
    dicom_tp_map = assign_tp(unique_dates)                    # 't0'→date, …

    rows = []
    for vol, tp_name, date, kernel in series_info:
        # which original t* column?
        orig_cols = {"t0": "", "t1": "", "t2": ""}
        for k, n in orig_tp_map.items():
            if n == tp_name:
                orig_cols[k] = str(vol)
                break

        # ── dicom_t* columns (by StudyDate order) ──────────────────────
        dicom_cols = {"dicom_t0": "", "dicom_t1": "", "dicom_t2": ""}
        for k, d in dicom_tp_map.items():
            if d == date:
                dicom_cols[f"dicom_{k}"] = str(vol)
                break

        rows.append({
            "pid": pid,
            **orig_cols,
            "filter": map_kernel(kernel),         # NLST code 1-12 / M
            **dicom_cols,
            "dicom_filter": kernel,               # raw text
        })
    return rows

# ────────────────────────────────────────────────────────────────────────
# 4.  Main
# ────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="root folder holding PID sub-dirs")
    ap.add_argument("csv_out", help="output CSV file")
    ap.add_argument("--min_slices", type=int, default=20,
                    help="skip series with fewer slices than this (0 = keep all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"{root} not found or not a directory.")

    all_rows: List[Dict[str, str]] = []
    for pid_dir in tqdm(sorted(root.iterdir())):
        if pid_dir.is_dir():
            all_rows.extend(rows_for_pid(pid_dir))

    fieldnames = [
        "pid", "t0", "t1", "t2", "filter",
        "dicom_t0", "dicom_t1", "dicom_t2", "dicom_filter"
    ]
    with open(args.csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {args.csv_out}")


if __name__ == "__main__":
    main()
    counts = Counter()
    with open("nlst_index.csv") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            counts[row["dicom_filter"]] += 1
    print(counts)
