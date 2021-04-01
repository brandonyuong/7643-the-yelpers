#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 14:22:43 2021

@author: sashaqanderson
"""
import json
import pandas as pd

# download dataset from https://www.kaggle.com/yelp-dataset/yelp-dataset

#load reviews from downloaded json file
data_file = open("./data/yelp_academic_dataset_review.json")
data = []
for line in data_file:
    data.append(json.loads(line))
review_df = pd.DataFrame(data)
data_file.close()

# load business from downloaded json file
data_file = open("./data/yelp_academic_dataset_business.json")
data = []
for line in data_file:
    data.append(json.loads(line))
business_df = pd.DataFrame(data)
data_file.close()

# review_df.to_csv("./data/reviews.csv", index=False)
# business_df.to_csv("./data/business.csv", index=False)

# review_df = pd.read_csv("./data/reviews.csv")
# business_df = pd.read_csv("./data/business.csv")

review_df = review_df[['business_id', 'stars', 'text']]
business_df = business_df[['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'is_open', 'categories']]
business_df.columns = ['business_id', 'name', 'city', 'state', 'avg_stars', 'review_count',
       'is_open', 'categories']
# print(len(business_df)) #160470

business_df = business_df[business_df['categories'].notna()]

restaurants_df = business_df[business_df['categories'].str.contains("estaurants")]
# print(len(df_restaurants)) #50763

# we need the review dataset and merge it with the business dataset to get type of business, location, etc.
result = pd.merge(restaurants_df, review_df, on="business_id", how = "left")
print(type(result.stars[0]))

result = result[result['stars'].notna()]
# replace stars with 1 or 2 with 'neg'
# replace stars with 4 or 5 with 'pos'

result.loc[result['stars'] == 1, 'stars'] = 'neg'
result.loc[result['stars'] == 2, 'stars'] = 'neg'
result.loc[result['stars'] == 4, 'stars'] = 'pos'
result.loc[result['stars'] == 5, 'stars'] = 'pos'

print(result.loc[66])

result['text'] = result['text'].astype(str)
final_df = result.groupby(['business_id','stars'])['text'].apply(lambda x: ','.join(x)).reset_index()

# print(len(final_df)) #143009

print(final_df.business_id.loc[1])
print(final_df.stars.loc[0])
print(final_df.text.loc[0])
print()
print()
print()
print(final_df.business_id.loc[1])
print(final_df.stars.loc[1])
print(final_df.text.loc[1])
