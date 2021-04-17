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

humorous = df[df['humor'] == 1]
not_humorous = df[df['humor'] == 0]

humorous = humorous.sample(n=200, random_state=8)
print(humorous)

not_humorous = not_humorous.sample(n=200, random_state=8)

print(humorous.shape[0])
print(humorous[humorous['word_count'] > 40].shape[0])
print(not_humorous.shape[0])
print(not_humorous[not_humorous['word_count'] > 40].shape[0])

df = pd.concat([humorous, not_humorous], ignore_index=True, axis=0)
# df = df[df['word_count'] > 40]

print(df.shape[0])

df.to_csv('./data/yelp_reduced.csv', index=False)
