"""Hold-out split, done once, fixed seed. Stratified on response x batch so both are balanced.
Test set is touched only at the very end (step 12). Everything else uses train_ids()."""
import pandas as pd
from sklearn.model_selection import train_test_split
PATH, TEST_FRAC, SEED = "results/split.csv", 0.2, 0

def make():
    df = pd.read_csv("Ornish.csv", index_col=0)
    strat = df["WeightLoss"] + "_" + pd.Series(df.index.str[:7], index=df.index)
    tr, te = train_test_split(df.index.to_numpy(), test_size=TEST_FRAC, stratify=strat, random_state=SEED)
    pd.DataFrame({"set": ["train"] * len(tr) + ["test"] * len(te)}, index=list(tr) + list(te)).to_csv(PATH)

def ids(which):
    return pd.read_csv(PATH, index_col=0).query("set == @which").index

if __name__ == "__main__":
    make(); s = pd.read_csv(PATH, index_col=0); wl = pd.read_csv("Ornish.csv", index_col=0)["WeightLoss"].loc[s.index]
    print(pd.crosstab(s.set, [wl, s.index.str[:7]]))
