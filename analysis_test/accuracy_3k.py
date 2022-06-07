# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import metrics
import logging

df = pd.read_csv("../data/SC/train_3k.tsv", sep="\t")
gold_list = df["label"].tolist()

df_pred = pd.read_csv("3k_test_results.tsv", sep="\t", header=None)
pred_list = []

for row in df_pred.itertuples(index=False):
    if row[0] < row[1]:
        pred_list.append(1)
    elif row[0] > row[1]:
        pred_list.append(0)
    else:
        pred_list.append(0.5)

m = metrics.classification_report(
            gold_list, pred_list,  digits=4
        )
print(m)

same = []
diff = []
for i in range(len(gold_list)):
    if gold_list[i] != pred_list[i]:
        diff.append(i)
        logging.error("sent:{},db:{},bert96:{},cls0:{},cls1:{}".format(df.iloc[i, 0], gold_list[i], pred_list[i], df_pred.iloc[i,0], df_pred.iloc[i,1]))
    else:
        same.append(i)

hard_df = df.iloc[diff]
print(hard_df.info())
hard_df.to_csv("train_3k_hard.tsv", sep="\t", index=False) #913
same_df = df.iloc[same]
print(same_df.info())
same_df.to_csv("train_3k_same.tsv", sep="\t", index=False) #2087