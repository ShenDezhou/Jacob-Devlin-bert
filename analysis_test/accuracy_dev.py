# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import metrics
import logging

df = pd.read_csv("../data/SC/dev.tsv", sep="\t")
gold_list = df["label"].tolist()

df_pred = pd.read_csv("dev_results.tsv", sep="\t", header=None)
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

for i in range(len(gold_list)):
    if gold_list[i] != pred_list[i]:
        logging.error("sent:{},gold:{},pred:{}".format(df.iloc[i, 0], gold_list[i], pred_list[i]))

#               precision    recall  f1-score   support
#
#            0     0.9429    0.9429    0.9429        35
#            1     0.8667    0.8667    0.8667        15
#
#     accuracy                         0.9200        50
#    macro avg     0.9048    0.9048    0.9048        50
# weighted avg     0.9200    0.9200    0.9200        50