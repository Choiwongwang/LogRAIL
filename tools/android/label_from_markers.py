#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label logcat lines using LOGRAIL_SCENARIO START/END markers.

Input:  raw logcat file (threadtime format)
Output: CSV with parsed fields and label (0/1)
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
    ap.add_argument("--in", dest="inp", required=True, help="Input logcat_raw.txt")
    ap.add_argument("--out", required=True, help="Output CSV with label column")
    ap.add_argument("--label_markers", action="store_true", default=True,
                    help="Label marker lines as anomaly (default: True)")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    in_anom = False
    cur_id = ""
    cur_name = ""

    with inp.open("r", encoding="utf-8", errors="ignore") as f_in, \
         out.open("w", newline="", encoding="utf-8") as f_out:
        wr = csv.writer(f_out)
        wr.writerow([
            "Date", "Time", "Pid", "Tid", "Level", "Tag", "Content",
            "label", "scenario_id", "scenario_name", "raw"
        ])

        for line in f_in:
            line = line.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue

            date = m.group("date")
            time = m.group("time")
            pid = m.group("pid")
            tid = m.group("tid")
            level = m.group("level")
            tag = m.group("tag").strip()
            msg = m.group("msg")

            marker = MARKER_RE.search(line)
            if marker:
                kind, sid, sname = marker.group(1), marker.group(2), marker.group(3)
                if kind == "START":
                    in_anom = True
                    cur_id, cur_name = sid, sname
                else:
                    in_anom = False
                    cur_id, cur_name = "", ""

            label = 1 if (in_anom or (marker and args.label_markers)) else 0
            wr.writerow([date, time, pid, tid, level, tag, msg, label, cur_id, cur_name, line])

    print(f"[✓] Labeled CSV saved → {out}")


if __name__ == "__main__":
    main()
