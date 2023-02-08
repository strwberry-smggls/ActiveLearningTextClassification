import pandas as pd
import sys
import os
from tqdm.auto import tqdm
import numpy as np
from collections import Counter

path = sys.argv[1]

files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]


label_per_sent = []
sents_per_label = {}
for f in files:
    print(f)
    df = pd.read_csv(f)
    print(len(df))
    for index, row in tqdm(df.iterrows()):
        labels = row.iloc[2:]
        n = 0
        i = 0
        for l in labels:
            label = int(l)
            n += label
            if (label == 1) and ("train" in f):
                if i not in sents_per_label:
                    sents_per_label[i] = 1
                else:
                    sents_per_label[i] += 1
            i += 1
        label_per_sent.append(n)
print(np.mean(label_per_sent))
print(np.mean(list(sents_per_label.values())))
print(len(df.columns))

mode = path.split("/")[-2]
with open(f"datasets/class_dist_{mode}.txt", "w") as f:
    c = sorted(list(Counter(sents_per_label).items()), key=lambda x: x[1], reverse=True)
    for v in c:
        f.write(f"{v[0]},{v[1]}\n")


# label correlation
label_map = {}
for f in files:
    print(f)
    df = pd.read_csv(f)
    print(len(df))
    for index, row in tqdm(df.iterrows()):
        labels = row.iloc[2:]
        for i in range(len(labels)):
            if i not in label_map:
                label_map[i] = {}
            l = labels[i]
            if l == 1:
                for j in range(len(labels)):
                    k = labels[j]
                    if i == j:
                        continue
                    if j not in label_map[i]:
                        label_map[i][j] = 0
                    if k == 1:
                        label_map[i][j] += 1
average_cooc = []
for k in label_map:
    # print(label_map[k].values())
    k_cooc_avg = np.mean(list(label_map[k].values()))
    if np.isnan(k_cooc_avg):
        k_cooc_avg = 0
    average_cooc.append(k_cooc_avg)


print(np.mean(average_cooc))
print(np.std(average_cooc))
print(max(average_cooc))
print(min(average_cooc))
