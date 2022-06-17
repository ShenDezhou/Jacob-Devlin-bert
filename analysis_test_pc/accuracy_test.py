# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn import metrics
import logging

tags = open("topic_all.dic",'r', encoding='utf-8').read().split()
exact_type = [0,1,2,6,7,9,10,12,15,16,17,18,19,23,26,30,37] + [3,8,14,20,33,38,39]
first_map = sorted(exact_type)
df = pd.read_csv("../data/PC/test.tsv", sep="\t")
gold_list = df["tag"].tolist()

df_pred = pd.read_csv("test_results.tsv", sep="\t", header=None)
pred_list = []

for row in df_pred.itertuples(index=False):
    pred_list.append(row.index(max(row)))


m = metrics.classification_report(
            gold_list, pred_list,  digits=4
        )
print(m)

for i in range(len(gold_list)):
    if gold_list[i] != pred_list[i]:
        logging.error("sent:{},gold:{},pred:{},gold-name:{},pred-name:{}".format(df.iloc[i, 0], gold_list[i], pred_list[i], tags[first_map[gold_list[i]]], tags[first_map[pred_list[i]]]))

#               precision    recall  f1-score   support
#
#            1     0.6364    1.0000    0.7778         7
#            2     0.9167    0.7857    0.8462        28
#            3     0.9231    0.6000    0.7273        20
#            6     0.3636    0.3077    0.3333        13
#            7     0.9677    1.0000    0.9836        30
#            8     1.0000    1.0000    1.0000         8
#            9     0.8276    0.8276    0.8276        29
#           10     0.2500    0.3333    0.2857         3
#           11     0.8571    0.8276    0.8421        29
#           12     0.8182    0.9474    0.8780        19
#           13     1.0000    1.0000    1.0000         6
#           14     0.9667    1.0000    0.9831        29
#           15     1.0000    1.0000    1.0000        10
#           16     0.4167    0.4167    0.4167        12
#           17     0.8182    0.9000    0.8571        30
#           18     0.3333    1.0000    0.5000         1
#           19     0.7500    0.6923    0.7200        26
#           20     0.4000    0.6667    0.5000         3
#           21     1.0000    1.0000    1.0000        30
#           22     0.4000    0.6667    0.5000         6
#           23     0.0000    0.0000    0.0000         5
#
#     accuracy                         0.8198       344
#    macro avg     0.6974    0.7606    0.7133       344
# weighted avg     0.8212    0.8198    0.8152       344