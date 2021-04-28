import json
import pandas as pd
import numpy as np
import re

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


RANDOM_STATE = 77
# SAMPLING_SIZE = 20000

df = pd.read_csv('./data/yelp_merged_data.csv')
print(df.head(10))

# df = df[df['word_count'] < 800]

# humorous = df[df['humor'] == 1]
humorous = df[df['funny'] >= 10]
not_humorous = df[df['humor'] == 0]

# print(humorous.shape[0])
SAMPLING_SIZE = humorous.shape[0]
print(SAMPLING_SIZE)

humorous = humorous.sample(n=SAMPLING_SIZE, random_state=RANDOM_STATE)
# print(humorous)

not_humorous = not_humorous.sample(n=SAMPLING_SIZE, random_state=RANDOM_STATE)

print(humorous.shape[0])
# print(humorous[humorous['word_count'] > 40].shape[0])
print(not_humorous.shape[0])
# print(not_humorous[not_humorous['word_count'] > 40].shape[0])

df = pd.concat([humorous, not_humorous], ignore_index=True, axis=0)

print(df.shape[0])

df.to_csv('./data/yelp_reduced_40k_50-50-humor-split.csv', index=False)
