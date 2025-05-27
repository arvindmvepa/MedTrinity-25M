from collections import Counter, defaultdict
import argparse, csv, os, re, sys
from pathlib import Path
from typing import Dict, List
import pydicom

# ----------------------------------------------------------------------
# NLST kernel-code lookup  (same as before + recent patches)
# ----------------------------------------------------------------------
FILTER_MAP = [
    # Siemens
    (re.compile(r"b50f?",        re.I), "7"),
    (re.compile(r"b2\d+f?",      re.I), "8"),  # B20-B29
    (re.compile(r"b3\d+f?",      re.I), "8"),  # B30-B39
    (re.compile(r"b7\d+f?",      re.I), "9"),  # B70-B79
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

def map_kernel(kernel_text: str) -> str:
    for pat, code in FILTER_MAP:
        if pat.search(kernel_text):
            return code
    return MISSING_CODE

# ----------------------------------------------------------------------
def first_dicom_in(d: Path) -> Path:
    """Return path of the first file in *d* that looks like DICOM."""
    for f in d.iterdir():
        if f.is_file() and (f.suffix.lower() == ".dcm" or f.suffix == ""):
            return f
    raise FileNotFoundError(f"No DICOM files in {d}")

def series_meta(series_dir: Path):
    """Return (StudyDate ISO str or '', ConvolutionKernel str) for a series."""
    ds = pydicom.dcmread(first_dicom_in(series_dir),
                         stop_before_pixels=True, force=True)

    date = (ds.get("StudyDate") or
            ds.get("SeriesDate") or
            ds.get("AcquisitionDate") or
            "").strip()
    kernel = ds.get("ConvolutionKernel", "").strip()
    return date, kernel

def assign_tp(values_sorted: List[str]) -> Dict[str, str]:
    """Map first three items to t0/t1/t2; return possibly sparse dict."""
    tp = {}
    for i, v in enumerate(values_sorted[:3]):
        tp[f"t{i}"] = v
    return tp

# ----------------------------------------------------------------------
def build_rows(pid_dir: Path) -> List[Dict[str, str]]:
    """
    Build CSV rows for **one PID**.
    """
    pid = pid_dir.name
    uid_dirs = [d for d in sorted(pid_dir.iterdir()) if d.is_dir()]
    if not uid_dirs:
        return []

    # ── original (folder-order) time-points ────────────────────────────
    orig_tp_map = assign_tp([d.name for d in uid_dirs])

    # ── collect per-series DICOM metadata ──────────────────────────────
    metas = []
    for udir in uid_dirs:
        try:
            date, kernel = series_meta(udir)
        except Exception as e:
            print(f"⚠️  {udir} skipped ({e})", file=sys.stderr)
            continue
        metas.append((udir, date, kernel))

    # derive DICOM-based tp mapping (by unique StudyDate)
    unique_dates = sorted({d for _, d, _ in metas if d})
    dicom_tp_map = assign_tp(unique_dates)   # 't0'→date, …

    rows = []
    for udir, date, kernel in metas:
        # pick which original tp column this UID occupies
        orig_cols = {"t0": "", "t1": "", "t2": ""}
        for k, uid_name in orig_tp_map.items():
            if uid_name == udir.name:
                orig_cols[k] = str(udir)
                break

        # pick which dicom_t* column this date occupies
        dt_cols = {"dicom_t0": "", "dicom_t1": "", "dicom_t2": ""}
        for k, d in dicom_tp_map.items():
            if d == date:
                dt_cols[f"dicom_{k}"] = date
                break

        rows.append({
            "pid": pid,
            **orig_cols,
            "filter": map_kernel(kernel),       # code 1-12 / M
            **dt_cols,
            "dicom_filter": kernel,             # raw kernel text
        })
    return rows

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="root folder containing PID sub-dirs")
    ap.add_argument("csv_out", help="output CSV file")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"{root} not found or not a directory.")

    all_rows: List[Dict[str, str]] = []
    for pid_dir in sorted(root.iterdir()):
        if pid_dir.is_dir():
            all_rows.extend(build_rows(pid_dir))

    # ── write CSV ────────────────────────────────────────────────────
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
