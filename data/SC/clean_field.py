# -*- coding: UTF-8 -*-
import pandas as pd

val = ["train", "dev", "test"]

for v in val:
    df = pd.read_csv(v+".tsv", sep="\t")
    df.to_csv(v+"_c.tsv", sep="\t", index=False, columns=["comment_context","label"])