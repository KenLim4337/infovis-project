

import numpy as np
import pandas as pd
import os
import sys
import pickle
from datetime import date
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath('..'))
sns.set_theme(style='whitegrid')


df = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_review_fused.gzip',compression='gzip',index_col=[0])
df_categories = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_category_counts.gzip',compression='gzip',index_col=[0])


# Handle nulls
# Set rows with 0 hours per week to NA
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'hours_per_week']

df[df['hours_per_week'] == 0][days] = np.nan

#Fill all numeric with mode
df.select_dtypes(include='number').fillna(df.select_dtypes(include='number').median(), inplace=True)

#Fill remainder with mode
null_columns = df.isnull().sum()[df.isnull().sum() > 0].index

for col in null_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# After filling NAs and 0s, recalculate hours per week.
df['hours_per_week'] = df['Monday'] + df['Tuesday'] + df['Wednesday'] + df['Thursday'] + df['Friday'] + df['Saturday'] + df['Sunday']

# Categorize if business is open on the weekend (Saturday and Sunday both)
df['open_weekends'] = ((df['Saturday'] > 0) & (df['Sunday'] > 0))

print(df['stars'].mean())

# Categorize if business is 'highly rated' (Above average)
df['highly_rated'] = df['stars'] >= df['stars'].mean()

print(df.head(5).to_string())

# Line Plot
daily_hours = df[days + ['highly_rated']]

daily_hours = daily_hours.groupby('highly_rated').mean().T.drop(['hours_per_week'])

daily_hours.reset_index(inplace=True)

daily_hours_high = daily_hours.drop(False, axis=1)
daily_hours_high['Highly Rated'] = True

daily_hours_lo = daily_hours.drop(True, axis=1)
daily_hours_lo['Highly Rated'] = False

daily_hours = pd.concat([daily_hours_high, daily_hours_lo])

daily_hours[1].fillna(daily_hours[0], inplace=True)

daily_hours.drop([0], axis=1, inplace=True)

daily_hours.rename(columns={'index':'day',1:'hours'}, inplace=True)

daily_hours.reset_index(inplace=True)


plt.figure()

sns.lineplot(data=daily_hours, x='day', y='hours', hue='Highly Rated')

plt.title('Mean hours open per day for highly rated businesses against other businesses')

plt.tight_layout()

plt.xlabel('Day')

plt.ylabel('Hours Open')

plt.show()


# Bar-plot : stack, group
df_stars = df.groupby('highly_rated').sum()[['star_1', 'star_2', 'star_3', 'star_4', 'star_5']].T

df_stars[False] = MinMaxScaler().fit_transform(np.array(df_stars[False]).reshape(-1,1))
df_stars[True] = MinMaxScaler().fit_transform(np.array(df_stars[True]).reshape(-1,1))

df_stars.plot(kind='bar', stacked=True)

plt.title('Distribution of review ratings for highly rated and lowly rated businesses\n')

plt.tight_layout()

plt.xlabel('Review Rating')

plt.xticks(range(0,5),['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'])

plt.ylabel('Normalized Number of Reviews')

plt.legend(title='Highly Rated')

plt.show()


df_stars.plot(kind='bar')

plt.title('Distribution of review ratings for highly rated and lowly rated businesses\n')

plt.tight_layout()

plt.xlabel('Review Rating')

plt.xticks(range(0,5),['1 Star', '2 Star', '3 Star', '4 Star', '5 Star'])

plt.ylabel('Normalized Number of Reviews')

plt.legend(title='Highly Rated')

plt.show()


# Count-plot
sns.countplot(data=df, x='state', hue='is_open', order=df['state'].value_counts().index)

plt.title('Number of open and closed businesses in each recorded state')

plt.xlabel('State')
plt.ylabel('Count')

plt.legend(title='Business Open', labels=['Closed', 'Open'])

plt.show()


# Cat-plot
g = sns.catplot(x='state',
            col='is_open',
            hue='highly_rated',
            kind='count',
            data=df)

g.fig.axes[0].set_title('Closed Businesses')
g.fig.axes[1].set_title('Open Businesses')

g.legend.set_title('Highly Rated')

g.set_axis_labels('State','Count')

plt.suptitle('Count of highly rated open and closed businesses')

plt.tight_layout()

plt.show()


# Pie chart
def my_autopct(pct):
    return ('%.2f %%' % pct) if pct > 7.5 else ''


statecounts = df['state'].value_counts()
statecounts = statecounts[statecounts.values > 1]

plt.pie(statecounts, labels=statecounts.keys(), autopct=my_autopct)

plt.title('Proportion of restaurants coming from each state')

plt.show()



# Displot
g = sns.displot(df, x='est_business_age', hue='highly_rated')

plt.title('Estimated Business Age for open and closed businesses')

plt.xlabel('Business Age')

g.legend.set_title('Highly Rated')

plt.tight_layout()

plt.show()


# Pair plot
sns.pairplot(data=df[['highly_rated','stars','review_count','avg_review_age','hours_per_week','est_business_age']], hue='highly_rated')
plt.show()


# Heatmap
df_corr = df[['stars','review_count','avg_review_age','hours_per_week','est_business_age']]

sns.heatmap(data=df_corr.corr(method='spearman'))

plt.show()


# QQ-plot
from statsmodels.graphics.gofplots import qqplot

plt.figure()
qqplot(df['est_business_age'],line='s')
plt.title('Business Age')
plt.show()



# Kernel density estimate
sns.kdeplot(data=df,
            x='hours_per_week',
            hue='highly_rated',
            bw_adjust=5,
            cut=0,
            multiple='stack')
plt.title('Kernel Density Estimate for hours_per_week')
plt.show()



# Scatterplot with regression line
from sklearn import linear_model
regr = linear_model.LinearRegression()

X = np.array(df['avg_review_age'].values).reshape(-1,1)
Y = np.array(df['est_business_age'].values).reshape(-1,1)

regr = linear_model.LinearRegression()

regr.fit(X, Y)

Y_pred = regr.predict(X)

fig = plt.figure()

sns.scatterplot(data=df,
            x='avg_review_age',
            y='est_business_age',
            hue='highly_rated',
)

line_df = pd.DataFrame({'X':np.squeeze(X), 'Y_pred':np.squeeze(Y_pred)})

sns.lineplot(data=line_df, x='X',y='Y_pred', color='green', label='Regression Line')

plt.title('Scatterplot of est_business_age against avg_review_age split on highly_rated')

plt.show()


# Multivariate Boxplot
sns.boxplot(data=df[['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']])
plt.title('Boxplot for Hours Open in each day of the week')
plt.ylabel('Count')
plt.xlabel('Day')
plt.show()


# Violin Plot
sns.violinplot(data=df, x="stars", y="state")
plt.title('Average star rating distribution among states')
plt.show()


# Subplots: You need to provide subplots in your report that tell a story to a reader. Pick a method discussed in class.
fig, ax = plt.subplots(1, 2)

values_overall = [len(df[df['highly_rated'] == True]), len(df[df['highly_rated'] == False])]

values_ca = [len(df[(df['highly_rated'] == True) & (df['state'] == 'CA')]), len(df[(df['highly_rated'] == False) & (df['state'] == 'CA')])]

labels = ['Highly Rated', 'Below Average']

ax[0].pie(x=values_overall, labels=labels, autopct='%.2f%%')

ax[0].set_title('Overall')

ax[1].pie(x=values_ca, labels=labels, autopct='%.2f%%')

ax[1].set_title('CA')

fig.suptitle('Proportion of highly rated restaurants to below average restaurants')

plt.tight_layout()

fig.show()


fig, ax = plt.subplots(1, 3, figsize=(30,10))

df_corr = df[['stars','review_count','avg_review_age','hours_per_week','est_business_age']]

df_corr_logged = np.sign(df_corr) * np.log(abs(df_corr) + 1)

sns.heatmap(ax=ax[0], data=df_corr.corr())

ax[0].set_title('Data as is, pearson correlation')

sns.heatmap(ax=ax[1], data=df_corr_logged.corr())

ax[1].set_title('Log scaled data, pearson correlation')

sns.heatmap(ax=ax[2], data=df_corr.corr(method='spearman'))

ax[2].set_title('Data as is, spearman correlation')

fig.suptitle('Heatmaps for different correlations in the dataset')

plt.tight_layout()

plt.show()

