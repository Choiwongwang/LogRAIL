#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess logcat:
1) Label lines using LOGRAIL_SCENARIO markers.
2) Remove marker lines into a clean text file.

Input:  raw logcat (threadtime)
Output: labeled CSV + cleaned TXT
"""

import argparse
import csv
import re
from pathlib import Path


LINE_RE = re.compile(
    r"^(?P<date>\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d{3})\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<tid>\d+)\s+"
    r"(?P<level>[VDIWEF])\s+"
    r"(?P<tag>[^:]+):\s+"
    r"(?P<msg>.*)$"
)

MARKER_RE = re.compile(r"LOGRAIL_SCENARIO:\s+(START|END)\s+id=(\d+)\s+name=([^\s]+)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input logcat_raw_merged.txt")
    ap.add_argument("--labeled_csv", required=True, help="Output CSV with label column")
    ap.add_argument("--clean_txt", required=True, help="Output TXT without marker lines")
    ap.add_argument("--label_markers", action="store_true", default=True,
                    help="Label marker lines as anomaly (default: True)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out_csv = Path(args.labeled_csv)
    out_txt = Path(args.clean_txt)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    in_anom = False
    cur_id = ""
    cur_name = ""

    with inp.open("r", encoding="utf-8", errors="ignore") as f_in, \
         out_csv.open("w", newline="", encoding="utf-8") as f_csv, \
         out_txt.open("w", encoding="utf-8") as f_txt:
        wr = csv.writer(f_csv)
        wr.writerow([
            "Date", "Time", "Pid", "Tid", "Level", "Tag", "Content",
            "label", "scenario_id", "scenario_name", "raw"
        ])

        for line in f_in:
            raw_line = line.rstrip("\n")
            m = LINE_RE.match(raw_line)
            if not m:
                continue

            marker = MARKER_RE.search(raw_line)
            if marker:
                kind, sid, sname = marker.group(1), marker.group(2), marker.group(3)
                if kind == "START":
                    in_anom = True
                    cur_id, cur_name = sid, sname
                else:
                    in_anom = False
                    cur_id, cur_name = "", ""

            label = 1 if (in_anom or (marker and args.label_markers)) else 0

            wr.writerow([
                m.group("date"),
                m.group("time"),
                m.group("pid"),
                m.group("tid"),
                m.group("level"),
                m.group("tag").strip(),
                m.group("msg"),
                label,
                cur_id,
                cur_name,
                raw_line,
            ])

            # Write to clean text only if not a marker line
            if not marker:
                f_txt.write(raw_line + "\n")

    print(f"[✓] Labeled CSV saved → {out_csv}")
    print(f"[✓] Clean TXT saved → {out_txt}")


if __name__ == "__main__":
    main()
