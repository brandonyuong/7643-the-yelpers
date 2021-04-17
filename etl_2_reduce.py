import json
import pandas as pd
import numpy as np
import re

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


df = pd.read_csv('./data/yelp_merged_data.csv')
print(df.head(10))

df = df.sample(n=125, random_state=2)
print(df)

# df = df[df['state'].str.contains("GA")]
# print(df['word_count'].mean()) # 111.2266
# print(df['word_count'].min()) # 1
# print(df['word_count'].max()) # 1002
df = df[df['word_count'] > 40]
# print(df.head(10))
# print(df.shape[0]) # 608937
# print(df['word_count'].mean()) # 136.5856
# print(df['word_count'].min()) # 41
# print(df['word_count'].max()) # 1002

df.to_csv('./data/yelp_reduced.csv', index=False)
