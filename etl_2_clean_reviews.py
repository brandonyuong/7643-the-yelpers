import json
import pandas as pd
import numpy as np
import re

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def word_count(review):
    """
    Here we are removing the spaces from start and end, and breaking every word whenever
    we encounter a space and storing them in a list. The len of the list is the total
    count of words.
    :param review:
    :return: (int) word count of string input
    """
    return len(str(review).strip().split(" "))


def format_review(review):
    """
    Use Regex to format reviews
    :param review:
    :return: (string) formatted review
    """
    string_review = str(review)
    string_review = re.sub("\n+", " ", string_review)  # replace \n
    string_review = re.sub(",{2,}", ",", string_review)  # replace 2 or more commas
    string_review = re.sub(" {2,}", " ", string_review)  # replace 2 or more spaces
    return string_review


df = pd.read_csv('./data/yelp_merged_data.csv')
print(df.head(10))

df = df[df['state'].str.contains("GA")]
df = df[df['text'].apply(lambda x: word_count(str(x)) > 40)]
df['text'] = df['text'].apply(format_review)
print(df.head(10))
# print(len(df)) # 5574795

df.to_csv('./data/yelp_cleaned_reviews.csv', index=False)
