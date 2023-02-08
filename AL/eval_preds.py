import os
import pickle
import sys
import pandas as pd

files = [f for f in os.listdir(sys.argv[1]) if f.endswith("preds.p")]
files = sorted(files, key=lambda x: int(x.split("_")[0]))

class_freq = []
with open(sys.argv[2], "r") as f:
    for line in f.read().split("\n"):
        line = line.split(",")
        if len(line) < 2:
            continue
        class_freq.append(int(line[0]))

label_acc = {}
for f in files:
    label_acc[f] = {}
    eval_data = pickle.load(open(os.path.join(sys.argv[1], f), "rb"))
    pred = eval_data[0]
    gold = eval_data[1]
    threshold = eval_data[2]

    pred_t = []
    for p in pred:
        pred_ = []
        for p_ in p:
            if p_ <= threshold:
                pred_.append(0)
            else:
                pred_.append(1)
        pred_t.append(pred_)

    for i in range(len(pred_t)):
        p = pred_t[i]
        g = gold[i]

        for j in range(len(p)):
            if j not in label_acc[f]:
                label_acc[f][j] = {"tp": 0, "fp": 0, "fn": 0}
            p_ = p[j]
            g_ = g[j]
            if (p_ == 1) and (g_ == 1):
                label_acc[f][j]["tp"] += 1
            elif (p_ == 1) and (g_ == 0):
                label_acc[f][j]["fp"] += 1
            elif (p_ == 0) and (g_ == 1):
                label_acc[f][j]["fn"] += 1

    label_to_f1 = []
    for label in label_acc[f]:
        tp = label_acc[f][label]["tp"]
        fp = label_acc[f][label]["fp"]
        fn = label_acc[f][label]["fn"]
        if tp == 0 and fp == 0:
            prec = 0
        else:
            prec = tp / (tp + fp)

        if tp == 0 and fn == 0:
            rec = 0
        else:
            rec = tp / (tp + fn)
        if prec == 0 or rec == 0:
            f1 = 0
        else:
            f1 = 2 * prec * rec / (prec + rec)

        label_acc[f][label]["f1"] = f1
        label_to_f1.append((label, f1))
    label_to_f1 = sorted(label_to_f1, key=lambda x: x[1], reverse=True)

    f_name = f.split("_")[0]
    n = len(label_to_f1)
    avg_freq_rank = 0
    avg_f1 = 0
    for i in range(n):
        if label_to_f1[i][1] < 0.5:
            continue
        freq_rank = class_freq.index(label_to_f1[i][0])
        avg_freq_rank += freq_rank
        avg_f1 += label_to_f1[i][1]
    avg_f1 = avg_f1 / n
    avg_freq_rank = avg_freq_rank / n

    print(f"{f_name},{avg_freq_rank},{avg_f1}")

    # print(f"{class_freq.index(label_to_f1[i][0])},{label_to_f1[i][1]}")
