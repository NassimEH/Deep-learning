import csv
import glob

for path in sorted(glob.glob("results_*.csv")):
    with open(path) as f:
        rows = list(csv.reader(f))
    if not rows:
        continue
    print(f"\n### {path}\n")
    print("| " + " | ".join(rows[0]) + " |")
    print("|" + "---|" * len(rows[0]))
    for r in rows[1:]:
        print("| " + " | ".join(r) + " |")
