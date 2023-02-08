import pandas as pd
import sys

train_df = pd.read_csv(sys.argv[1]+"train.csv")
test_df = pd.read_csv(sys.argv[1]+"test.csv")

datatsets = [train_df, test_df]
names = ["lm_train.txt", "lm_test.txt"]

for i in range(len(datatsets)):
    df = datatsets[i]
    name = names[i]
    outtext = ""
    for t in df["comment_text"]:
        outtext += t
        outtext += "\n"
    outtext = outtext.strip("\n")
    with open(sys.argv[1]+name, "w") as f:
        f.write(outtext)
