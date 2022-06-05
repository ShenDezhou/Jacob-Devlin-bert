# -*- coding: UTF-8 -*-
import pandas as pd
df = pd.read_csv("original/comment_high_value_2021_03_05_2022_04_05.csv", encoding="utf-8")
df["label"] = ""
print(df.info())
for i in range(df.shape[0]):
    comment = df.iloc[i, 1]
    if '喜欢' in comment:
        df.iloc[i, 3] = "1"
    elif '不' in comment:
        df.iloc[i, 3] = "0"
    else:
        df.iloc[i, 3] = ""

train = df.sample(n=300)
dev = df.sample(n=50)
test = df.sample(n=100)
train.to_csv("train.tsv", sep='\t', encoding="utf-8")
dev.to_csv("dev.tsv", sep='\t', encoding="utf-8")
test.to_csv("test.tsv", sep='\t', encoding="utf-8")