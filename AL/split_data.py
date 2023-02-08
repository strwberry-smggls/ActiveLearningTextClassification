import pandas
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

sents = pandas.read_csv(
    "datasets/stanfordSentimentTreebank/datasetSentences.txt",
    sep="\t",
    index_col=["sentence_index"],
)
split_index = pandas.read_csv(
    "datasets/stanfordSentimentTreebank/datasetSplit.txt",
    sep=",",
    index_col=["sentence_index"],
)
sentiment = pandas.read_csv(
    "datasets/stanfordSentimentTreebank/sentiment_labels.txt",
    sep="|",
    index_col=["phrase ids"],
)
phrases = pandas.read_csv("datasets/stanfordSentimentTreebank/dictionary.txt", sep="|")


def map_label(df_dict, label):
    if 0 < label <= 0.2:
        df_dict["l1"].append(1)
        df_dict["l2"].append(0)
        df_dict["l3"].append(0)
        df_dict["l4"].append(0)
        df_dict["l5"].append(0)
    elif 0.2 < label <= 0.4:
        df_dict["l1"].append(0)
        df_dict["l2"].append(1)
        df_dict["l3"].append(0)
        df_dict["l4"].append(0)
        df_dict["l5"].append(0)
    elif 0.4 < label <= 0.6:
        df_dict["l1"].append(0)
        df_dict["l2"].append(0)
        df_dict["l3"].append(1)
        df_dict["l4"].append(0)
        df_dict["l5"].append(0)
    elif 0.6 < label <= 0.8:
        df_dict["l1"].append(0)
        df_dict["l2"].append(0)
        df_dict["l3"].append(0)
        df_dict["l4"].append(1)
        df_dict["l5"].append(0)
    elif 0.8 < label <= 1.0:
        df_dict["l1"].append(0)
        df_dict["l2"].append(0)
        df_dict["l3"].append(0)
        df_dict["l4"].append(0)
        df_dict["l5"].append(1)


train_df = {"sents": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []}
test_df = {"sents": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []}
val_df = {"sents": [], "l1": [], "l2": [], "l3": [], "l4": [], "l5": []}
sentiment_dict = {}
# print(sents.head())
for sent in tqdm(sents.iterrows()):
    s = sent[1][0]
    id = sent[0]
    sentiments = []
    for phrase in tqdm(phrases.iterrows()):
        p_id = phrase[0]
        p = phrase[1][0]
        if p in s:
            sentiments.append(sentiment.loc[p_id]["sentiment values"])
    split = split_index.loc[id]["splitset_label"]
    label = np.average(sentiments)

    if split == 1:
        train_df["sents"].append(s)
        map_label(train_df, label)
    elif split == 2:
        test_df["sents"].append(s)
        map_label(test_df, label)
    elif split == 3:
        val_df["sents"].append(s)
        map_label(val_df, label)
train_df = pandas.DataFrame.from_dict(train_df)
test_df = pandas.DataFrame.from_dict(test_df)
val_df = pandas.DataFrame.from_dict(val_df)

train_df.to_csv("sst_train.csv", index=False)
test_df.to_csv("sst_test.csv", index=False)
val_df.to_csv("sst_val.csv", index=False)


"""
df = pandas.read_csv("datasets/toxic_col/train.csv")
print(len(df))
train, dev = train_test_split(df, test_size=0.2)
train, test = train_test_split(train, test_size=0.2)

train.to_csv("train.csv", index=False)
dev.to_csv("dev.csv", index=False)
test.to_csv("test.csv", index=False)
"""
