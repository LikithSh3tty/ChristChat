"""Re-apply the current scoring rules to saved answers without re-running the LLMs.

Use after editing experiments/questions.yaml so all result files are scored by the same rules.
Run:  python -m experiments.rescore experiments/results/model1_closed_book.csv [more.csv ...]
"""
import sys

import pandas as pd

from common.scoring import load_questions, score

if __name__ == "__main__":
    qs = {q["id"]: q for q in load_questions()}
    for path in sys.argv[1:]:
        df = pd.read_csv(path)
        df["text"] = df["text"].fillna("")
        s = pd.DataFrame([score(qs[i], t) for i, t in zip(df["id"], df["text"])], index=df.index)
        df[s.columns] = s
        df.to_csv(path, index=False)
        print(path, df.groupby("model")["correct"].mean().round(3).to_dict())
