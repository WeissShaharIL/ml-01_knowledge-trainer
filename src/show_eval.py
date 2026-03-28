import json, os, glob

pattern = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_round_*.json")
files   = sorted(glob.glob(pattern))
path    = files[-1]
print(f"Reading: {os.path.basename(path)}\n")
d = json.load(open(path))
for i, r in enumerate(d, 1):
    if not r["is_correct"]:
        print(f"[{i}] WRONG")
        print(f"  Q: {r['question']}")
        print(f"  A (correct): {r['correct_answer']}")
        print(f"  A (student): {r['student_answer']}")
        print()