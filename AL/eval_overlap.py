import os
import sys
import pandas as pd

path1 = "outputs_warmstart/eurlex_savedpreds/subword_run1"
random = "outputs_warmstart/eurlex_savedpreds/alps_run1"

files1 = sorted(
    [f for f in os.listdir(path1) if f.endswith("train.csv")],
    key=lambda x: int(x.split("_")[0]),
)
files_random = sorted(
    [f for f in os.listdir(random) if f.endswith("train.csv")],
    key=lambda x: int(x.split("_")[0]),
)

files1 = files1[: min(len(files1), len(files_random))]
files_random = files_random[: min(len(files1), len(files_random))]

for i in range(len(files1)):
    f = files1[i]
    rand_f = files_random[i]
    df = pd.read_csv(os.path.join(path1, f))
    df_random = pd.read_csv(os.path.join(random, rand_f))
    docs = []
    docs_rand = []
    labels = []
    labels_rand = []

    for j in range(len(df)):
        row = df.iloc[j, :]
        row_rand = df_random.iloc[j, :]
        docs.append(row[0])
        docs_rand.append(row_rand[0])
        label_inds = [k for k in range(len(row[2:])) if row[k] == 1]
        labels.extend(label_inds)

    docs = set(docs)
    docs_rand = set(docs_rand)
    print(
        f"{f.strip('_train.csv')},{len(docs.intersection(docs_rand))-100},{len(set(labels))/len(row[2:])}"
    )
