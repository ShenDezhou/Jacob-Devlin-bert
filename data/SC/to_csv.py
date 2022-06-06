# -*- coding: UTF-8 -*-
import pandas as pd

val = ["train", "dev", "test"]

for v in val:
    df = pd.read_csv(v+".tsv", sep="\t")
    df.to_excel(v+".xls", index=False, columns=["comment_context","label"])