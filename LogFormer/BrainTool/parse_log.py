# brain.py

import os
import sys
import time
import threading

# Add current file directory to sys.path.
current_dir = os.path.abspath(os.path.dirname(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import LogParser class from parse.py.
from Brain import LogParser

# -----------------------------------------------------------------------------
# Set input/output directories and log filename
# -----------------------------------------------------------------------------
input_dir = r'/home/irv4/HybridTransformer/DataFile'
output_dir = r'/home/irv4/HybridTransformer/DataFile'
log_file = 'anormal_logs.txt'  # actual log filename

# -----------------------------------------------------------------------------
# Android log format string (for arg shape; not used in parse)
# -----------------------------------------------------------------------------
android_format = "<Date> <Time>  <Level>/<Component>(<Pid>): <Content>"

# -----------------------------------------------------------------------------
# Regex patterns for parameter substitution
# -----------------------------------------------------------------------------
android_regex = [
    r"(/[\w-]+)+",                        # file path pattern example
    r"([\w-]+\.){2,}[\w-]+",             # domain pattern example
    r"\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b",  # number/hex
]

# -----------------------------------------------------------------------------
# Parsing parameters
# -----------------------------------------------------------------------------
threshold = 5       # Similarity threshold for Android logs
delimeter = []      # Delimiters for Android logs (unused here)

def start_heartbeat(interval_sec=60):
    """
    Heartbeat thread prints 'still parsing' periodically.
    """
    def beat():
        elapsed = 0
        while True:
            time.sleep(interval_sec)
            elapsed += interval_sec
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] still parsing... {elapsed//60} min {elapsed%60} sec elapsed")
    t = threading.Thread(target=beat, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    # Set format/regex/dataset name
    # -----------------------------------------------------------------------------
    log_format = android_format
    rex = android_regex
    dataset = 'Android'

    # -----------------------------------------------------------------------------
    # Create parser
    # -----------------------------------------------------------------------------
    parser = LogParser(
        logname=dataset,
        log_format=log_format,
        indir=input_dir,
        outdir=output_dir,
        threshold=threshold,
        delimeter=delimeter,
        rex=rex
    )

    # -----------------------------------------------------------------------------
    # Log start parsing
    # -----------------------------------------------------------------------------
    start_time = time.time()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Parsing started: {log_file}")

    # -----------------------------------------------------------------------------
    # Start heartbeat thread (print progress every 60s)
    # -----------------------------------------------------------------------------
    start_heartbeat(interval_sec=60)

    # -----------------------------------------------------------------------------
    # Run parsing
    # -----------------------------------------------------------------------------
    parser.parse(log_file)

    # -----------------------------------------------------------------------------
    # End parsing and print elapsed time
    # -----------------------------------------------------------------------------
    elapsed = time.time() - start_time
    finish_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{finish_timestamp}] Parsing finished: {log_file}")
    print(f"Elapsed time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} min, {elapsed:.1f} sec)")
