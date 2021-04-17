import json
import pandas as pd
import numpy as np
import re

# no truncation when printing df
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# download dataset from https://www.kaggle.com/yelp-dataset/yelp-dataset


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
    string_review = re.sub("\s+", " ", string_review)  # replace whitespace chars
    string_review = re.sub(",{2,}", ",", string_review)  # replace 2 or more commas
    string_review = re.sub(" {2,}", " ", string_review)  # replace 2 or more spaces
    return string_review


#load reviews from downloaded json file
data_file = open("./data/yelp_academic_dataset_review.json", encoding="utf8")
data = []
for line in data_file:
    data.append(json.loads(line))
review_df = pd.DataFrame(data)
data_file.close()

# load business from downloaded json file
data_file = open("./data/yelp_academic_dataset_business.json", encoding="utf8")
data = []
for line in data_file:
    data.append(json.loads(line))
business_df = pd.DataFrame(data)
data_file.close()
data = None

review_df = review_df[['business_id', 'funny', 'text']]
# print(review_df.shape[0]) # 8635403

business_df = business_df[
    ['business_id', 'name', 'city', 'state', 'categories']
]
business_df.columns = ['business_id', 'name', 'city', 'state', 'categories']
# print(business_df.shape[0]) # 160585

# review_df.dropna(inplace=True)
# print(review_df.shape[0]) # 8635403
business_df.dropna(subset=['categories'], inplace=True)
# print(business_df.shape[0]) # 160470

print(business_df.head(3))

business_df = business_df[business_df['categories'].str.contains("estaurants")]
merged_df = pd.merge(business_df, review_df, on="business_id", how="left")
merged_df['text'] = merged_df['text'].apply(format_review)
merged_df['word_count'] = merged_df['text'].apply(word_count)


# def get_sentiment(row):
#     # For use with pandas apply().  Make sure row['stars'] does not have NaN.
#     if row['stars'] < 3:
#         sentiment = -1
#     elif row['stars'] > 3:
#         sentiment = 1
#     else:
#         sentiment = 0
#     return sentiment


# merged_df['sentiment'] = merged_df.apply(get_sentiment, axis=1)

humor_threshold = 3


def eval_humor(row):
    # Evaluate humor based on "funny" column
    # For use with pandas apply()
    if row['funny'] >= humor_threshold:
        val = 1
    elif row['funny'] == 0:
        val = 0
    else:
        val = np.nan
    return val


merged_df['humor'] = merged_df.apply(eval_humor, axis=1)

print(merged_df.head(10))
# print(merged_df.shape[0]) # 5574795
# print(merged_df['word_count'].mean()) # 106.4523
# print(merged_df['word_count'].median()) # 76.0
# print(merged_df['word_count'].min()) # 1
# print(merged_df['word_count'].max()) # 1028
# print(merged_df['funny'].mean()) # 0.3851
# print(merged_df['funny'].median()) # 0.0
# print(merged_df['funny'].max()) # 610
#
# humor_label_pt = merged_df.pivot_table('text', ['humor'], aggfunc='count').reset_index()
# print(humor_label_pt)
"""
   humor     text
0    0.0  4582852
1    1.0   197795
"""

merged_df.to_csv('./data/yelp_merged_data.csv', index=False)
