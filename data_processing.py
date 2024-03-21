import numpy as np
import pandas as pd
import os
import sys
import pickle
from datetime import date
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import re

sys.path.append(os.path.abspath('..'))

df = pd.read_json('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/yelp_data/yelp_academic_dataset_business.json', lines=True)

# Only keep restaurants, the most popular category/just over 50k businesses
df = df[df['categories'].fillna('no').str.contains('Restaurant')]

# Drop address and attributes as it would require too much processing to be useful for visualization
df.drop(['address', 'attributes', 'postal_code'], axis=1, inplace=True)


# Remove states that only have less than 5 examples
statecounts = df['state'].value_counts()
statecounts = statecounts[statecounts.values < 5]
low_states = statecounts.index

df = df[~df['state'].isin(low_states)]


days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

def hours_to_columns(row):
    if row['hours'] is None:
        for day in days:
            row[day] = None
        return row
    else:
        for day in days:
            if row['hours'].get(day) is None:
                row[day] = 0
            else:
                hours = row['hours'].get(day).split('-')
                open_time = float(hours[0].split(':')[0]) + float(float(hours[0].split(':')[1])/60)
                close_time = float(hours[1].split(':')[0]) + float(float(hours[1].split(':')[1])/60)
                row[day] = abs(close_time - open_time)
        return row


# Process Business Data
df_processed = df.copy()

# Index of business IDs
business_ids = df_processed['business_id'].unique()

# Split hours into 7 columns denoting number of hours open a day
df_processed = df_processed.apply(hours_to_columns, axis=1)
df_processed.set_index('business_id', inplace=True)


# Split Business Categories and remove outlier categories (in terms of appearance from the dataset
categories = df_processed['categories'].astype(str).values

categories = ', '.join(categories)

categories = re.sub(r"(\s*?),", ", ", categories)

categories = re.split(', ', categories)

categories = [x.strip() for x in categories]

categories_unique = [*set(categories)]

categories_count = {}

for category in tqdm(categories_unique, desc='Counting categories'):
    categories_count[category] = categories.count(category)

# Parse category counts into a DF
df_categories = pd.DataFrame.from_dict(categories_count, orient='index', columns=['count'])

# Sort categories descending
df_categories = df_categories.sort_values(by='count', ascending=False)


df_categories.to_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_category_counts.gzip', compression='gzip')

# Processing Reviews
pd_reviews = pd.read_json('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/yelp_data/yelp_academic_dataset_review.json', lines=True, chunksize=10000)

today = date.today()

# Add empty columns to business table
df_processed['oldest_review'] = -sys.maxsize
df_processed['newest_review'] = sys.maxsize
df_processed['total_age'] = 0
df_processed['total_liked'] = 0
df_processed['star_1'] = 0
df_processed['star_2'] = 0
df_processed['star_3'] = 0
df_processed['star_4'] = 0
df_processed['star_5'] = 0
df_processed['review_dates'] = ''

count = 0
for chunk in tqdm(pd_reviews, desc='Processing review data'):
    temp = chunk.drop(['review_id', 'user_id', 'text'],axis=1)

    #Filter out rows that dont need to be there (non restaurants)
    temp = temp[temp['business_id'].isin(business_ids)]

    businesses = temp['business_id'].unique()

    # Age reviews
    temp['date'] = pd.to_datetime(temp['date'])
    temp['age'] = round((pd.to_datetime(today) - temp['date']).dt.days/365, 2)

    # Determine if a review is socially liked/supported by checking useful, funny, and cool columns
    temp['liked'] = temp[['useful', 'funny', 'cool']].any(axis=1) * 1

    # Get oldest review, newest review, age sum, number of liked, number for each star value for each business
    for business in businesses:
        temp_business = temp[temp['business_id'] == business]
        age_max = temp_business['age'].max()
        age_min = temp_business['age'].min()
        total_age = temp_business['age'].sum()
        total_liked = temp_business['liked'].sum()
        stars = temp_business['stars'].value_counts()
        dates = list(temp_business['date'].values)

        #Compare and update min/max
        if age_max > float(df_processed.at[business, 'oldest_review']):
            df_processed.at[business, 'oldest_review'] = age_max

        if age_min < float(df_processed.at[business, 'newest_review']):
            df_processed.at[business, 'newest_review'] = age_min

        #Add to total age and liked
        df_processed.at[business, 'total_age'] = float(df_processed.at[business, 'total_age']) + total_age
        df_processed.at[business, 'total_liked'] = float(df_processed.at[business, 'total_liked']) + total_liked
        df_processed.at[business, 'review_dates'] = list(df_processed.at[business, 'review_dates']) + dates

        #Star columns
        for key in stars.keys():
            df_processed.at[business, 'star_'+str(key)] = float(df_processed.at[business, 'star_'+str(key)]) + stars[key]

        count+=1

        if count == 3:
            break


# Derived columns: average review age, hours per week, estimated business age
df_processed['avg_review_age'] = round(df_processed['total_age']/df_processed['review_count'],2)
df_processed['est_business_age'] = round(abs(df_processed['oldest_review'] - df_processed['newest_review']),2)
df_processed['hours_per_week'] = df_processed['Monday'] + df_processed['Tuesday'] + df_processed['Wednesday'] + df_processed['Thursday'] + df_processed['Friday'] + df_processed['Saturday'] + df_processed['Sunday']

df_checkin = pd.read_json('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/yelp_data/yelp_academic_dataset_checkin.json', lines=True)

df_checkin.rename(columns={'date':'checkins'}, inplace=True)

df_processed = pd.merge(left=df_processed, right=df_checkin, how='inner', on='business_id')


df_processed.to_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_review_fused.gzip', compression='gzip')
