# -*- coding: UTF-8 -*-
import pandas as pd

exact_type = [0,1,2,6,7,9,10,12,15,16,17,18,19,23,26,30,37]
type_map = dict(zip(exact_type, range(len(exact_type))))


train = pd.read_csv("original/train_315.csv", encoding="utf-8")
train['tag'] = train['tag'].map(lambda i: type_map[i])
train.to_csv("train.tsv", sep='\t', columns=["sentence","tag"],index=False,  encoding="utf-8")

dev = pd.read_csv("original/dev_173.csv", encoding="utf-8")
dev['tag'] = dev['tag'].map(lambda i: type_map[i])
dev.to_csv("dev.tsv", sep='\t', index=False, encoding="utf-8")

test = pd.read_csv("original/test_282.csv", encoding="utf-8")
test['tag'] = test['tag'].map(lambda i: type_map[i])
test.to_csv("test.tsv", sep='\t',index=False,  encoding="utf-8")