#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:22:01 2021

@author: sashaqanderson
"""
import json
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# download dataset from https://www.kaggle.com/yelp-dataset/yelp-dataset

#load reviews from downloaded json file
def df_from_json(json_file):
    """
    Here we are taking one of the downloaded yelp json files and saving them as CSVs.
    :param review:
    :return: dataframe
    """
    data_file = open(json_file)
    data = []
    for line in data_file:
        data.append(json.loads(line))
    df = pd.DataFrame(data)
    data_file.close()
    # df.to_csv("./data/reviews.csv", index=False)
    return df
    
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

# csv_from_json("./data/yelp_academic_dataset_review.json", "./data/reviews.csv")
# csv_from_json("./data/yelp_academic_dataset_business.json", "./data/business.csv")
    

# review_df = pd.read_csv("./data/reviews.csv")
# review_df = review_df[['user_id', 'business_id', 'stars', 'useful', 'funny','text']]
# review_df['text'] = review_df['text'].apply(format_review)
# review_df['word_count'] = review_df['text'].apply(word_count)


# business_df = pd.read_csv("./data/business.csv")
# business_df = business_df[['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'is_open', 'attributes', 'categories']]
# business_df.columns = ['business_id', 'name', 'city', 'state', 'avg_stars', 'review_count','is_open', 'attributes','categories']
# business_df = business_df[business_df['name'].notna()]

# #take a subset of business in GA
# business_df_small = business_df[business_df['state'].str.contains("GA")]

# #merge reviews with business ids to get the name, etc.
# merged_data_large = pd.merge(business_df, review_df, on="business_id", how = "left")
# merged_data_small = pd.merge(business_df_small, review_df, on="business_id", how = "left")

# # print(len(merged_data_small)) #1150884
# # print(len(merged_data_large)) #8635403

# #concatenate the name of the business and the text together (some humor is based on the business name)
# merged_data_small['name_text'] = merged_data_small.name + '. ' + merged_data_small.text
# merged_data_large['name_text'] = merged_data_large.name + '. ' + merged_data_large.text

# temp_df_small = merged_data_small[['business_id', 'categories', 'user_id', 'funny', 'name_text', 'word_count']]
# temp_df_large = merged_data_large[['business_id', 'categories', 'user_id', 'funny', 'name_text', 'word_count']]


# # we will consider a review as humorous if it has 5 or more funny votes
# # and not humorous if it has 0 funny votes
# funny_threshold = 5
# humor_pt = temp_df_large.pivot_table('user_id', ['funny'], aggfunc='count').reset_index()
# funny_votes = humor_pt.funny.values

# plt.hist(funny_votes, bins=50, density = False)  # density=False would make counts
# plt.axvline(x=5, color = 'red')
# plt.text(5, 12, ' Funny Threshold = 5', color = 'pink')
# plt.ylabel('# Reviews')
# plt.xlabel('# Funny Votes')
# plt.title('Distribution of Funny Votes in Reviews')
# plt.show()

# final_df_small = temp_df_small.copy()
# final_df_large = temp_df_large.copy()

# def f(row):
#     if row['funny'] >= funny_threshold:
#         val = 1
#     elif row['funny'] == 0:
#         val = 0
#     else:
#         val = 'NaN'
#     return val

# final_df_small['humor'] = final_df_small.apply(f, axis=1)
# final_df_large['humor'] = final_df_small.apply(f, axis=1)

# humor_label_pt = final_df_large.pivot_table('business_id', ['humor'], aggfunc='count').reset_index()
# print(humor_label_pt)

#SMALL
#   humor  business_id
# 0     0       956553
# 1     1        16629
# 2   NaN       177702

#LARGE