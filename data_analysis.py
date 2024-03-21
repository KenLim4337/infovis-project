
import numpy as np
import pandas as pd
import os
import sys
import pickle
from datetime import date
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('..'))

sns.set_theme(style='whitegrid')

# Load data
df = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_review_fused.gzip',compression='gzip',index_col=[0])
df_categories = pd.read_csv('C:/Users/lkyoo/PycharmProjects/pythonProject1/infovis-project/processed_data/business_category_counts.gzip',compression='gzip',index_col=[0])

df['highly_rated'] = df['stars'] >= df['stars'].mean()


# Drop restaurants and food for categories as all remaining businesses are restaurants
df_categories.drop(['Restaurants', 'Food'], axis=0, inplace=True)

df_processed = df.copy()

#Further modification of dataset if necessary
df_processed.drop(['total_age'], axis=1, inplace=True)

#Round lat and long
df_processed['latitude'] = df_processed['latitude'].round(2)
df_processed['longitude'] = df_processed['longitude'].round(2)

# Before removing nulls
print(df_processed.isna().sum().sum())

# Missing columns are all hours, since all businesses being analyzed are restaurants (similar types of businesses), it is reasonable to take the mode/median to fill both hours and hours open per day. Also consider all hours per week = 0 businesses as NA for the purpose of this section as these are either missing or erroneous entries. Note: some businesses marked as closed have hours listed, therefore NAs are not simply closed businesses.

#Set rows with 0 hours per week to NA
days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'hours_per_week']

df_processed[df_processed['hours_per_week'] == 0][days] = np.nan

#Fill all numeric with mean
df_processed.select_dtypes(include='number').fillna(df_processed.select_dtypes(include='number').mean(), inplace=True)

#Fill remainder with mode
null_columns = df_processed.isnull().sum()[df_processed.isnull().sum() > 0].index

for col in null_columns:
    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# After filling NAs and 0s, recalculate hours per week.
df_processed['hours_per_week'] = df_processed['Monday'] + df_processed['Tuesday'] + df_processed['Wednesday'] + df_processed['Thursday'] + df_processed['Friday'] + df_processed['Saturday'] + df_processed['Sunday']

# Categorize if business is open on the weekend (Saturday and Sunday both)
df_processed['open_weekends'] = ((df_processed['Saturday'] > 0) & (df_processed['Sunday'] > 0))

# After removing nulls
print(df_processed.isna().sum().sum())

print(df_processed.head(5).to_string())

# Using IQR method
q1 = df_processed['hours_per_week'].quantile(.25)
q3 = df_processed['hours_per_week'].quantile(.75)
iqr = q3-q1
scale = 1.5

bottom_limit = q1 - (iqr*scale)
top_limit = q3 + (iqr*scale)

sns.boxplot(df_processed['hours_per_week'])
plt.grid(axis='x', which='minor')
plt.xlabel('Average Review Age')
plt.title('Boxplot of Average Review Age for Restaurants')
plt.axvline(bottom_limit, c='red', label="Bottom cutoff for outliers")
plt.axvline(top_limit, c='green', label="Top cutoff for outliers")
plt.legend()
plt.show()

# Remove outliers
df_processed = df_processed[(df_processed['hours_per_week'] <= top_limit) & (df_processed['hours_per_week'] >= bottom_limit)]


sns.boxplot((df_processed['hours_per_week']))
plt.grid(axis='x', which='minor')
plt.xlabel('Average Review Age')
plt.title('Boxplot of Average Review Age for Restaurants')
plt.axvline(bottom_limit, c='red', label="Bottom cutoff for outliers")
plt.axvline(top_limit, c='green', label="Top cutoff for outliers")
plt.legend()
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# All numeric features except is_open since its the label
X = df_processed.select_dtypes(include='number').drop('is_open', axis=1).values

# Scale features
X = StandardScaler().fit_transform(X)

# Keep all components for extra info
pca_full = PCA(n_components=len(df_processed.select_dtypes(include='number').drop('is_open', axis=1).columns), svd_solver='full')
X_pca_full = pca_full.fit_transform(X)
# MLE method
pca = PCA(n_components='mle', svd_solver='full')
X_pca = pca.fit_transform(X)

# Minimum to achieve >0.90
pca_min = PCA(n_components=8, svd_solver='full')
X_pca_min = pca_min.fit_transform(X)

# Make 'is_open' the dependent variable
Y = df_processed['is_open']

# Condition number for original data
print(f'Condition number for original data is {np.linalg.cond(X):.2f} \n')

# EVR, condition number and Single Values
print(f'Explained variance ratio for all principal components: {pca_full.explained_variance_ratio_.round(2)}')
print(f'Singular values for all principal components: {pca_full.singular_values_.round(2)}')
print(f'Condition number for PCA with all components  is {np.linalg.cond(X_pca_full):.2f}\n')

print(f'Explained variance ratio for principal components selected by MLE: {pca.explained_variance_ratio_.round(2)}')
print(f'Singular values for principal components selected by MLE: {pca.singular_values_.round(2)}')
print(f'Condition number for PCA with MLE is {np.linalg.cond(X_pca):.2f}\n')

print(f'Explained variance ratio for manually tuned principal components: {pca_min.explained_variance_ratio_.round(2)}')
print(f'Singular values for manually tuned principal components: {pca_min.singular_values_.round(2)}')
print(f'Condition number for manually tuned PCA is {np.linalg.cond(X_pca_min):.2f}\n')

plt.figure()
plt.plot(np.arange(1, len(np.cumsum(pca_full.explained_variance_ratio_))+1, 1), np.cumsum(pca_full.explained_variance_ratio_), label='Cumulative Sum of Explained Variance')
plt.axhline(0.90, c='red', label="0.90 Cumulative Explained Variance")
plt.legend()
plt.ylabel('Cumulative Sum of Explained Variance')
plt.xlabel('Number of Pricipal Components')
plt.title('Graph of Cumulative Explained Variance Over Number of Principal Components')
plt.xticks(np.arange(1, len(np.cumsum(pca_full.explained_variance_ratio_))+1, 1))
plt.tight_layout()
plt.show()

from scipy.stats import shapiro

# Get all numerical features sans is_open
df_norm = df_processed.select_dtypes(include='number').drop('is_open', axis=1)
alpha = 0.01

result = {}

for label, values in df_norm.iteritems():
    stats, p = shapiro(values)
    result[label] = {}
    if p > alpha:
        result[label]['normal'] = True
    else:
        result[label]['normal'] = False

    result[label]['stat'] = round(stats, 2)
    result[label]['p-value'] = p

df_norm = pd.DataFrame(result)

print(df_norm.T.to_string())

# Log everything and rerun shapiro?
target_cols = df_processed.select_dtypes(include='number').columns.drop('is_open')
df_norm = df_processed.copy()
df_norm[target_cols] = np.sign(df_norm[target_cols]) * np.log(abs(df_norm[target_cols]) + 1)

df_norm2 = df_norm.select_dtypes(include='number').drop('is_open', axis=1)
alpha = 0.01

result = {}

for label, values in df_norm2.iteritems():
    stats, p = shapiro(values)
    result[label] = {}
    if p > alpha:
        result[label]['normal'] = True
    else:
        result[label]['normal'] = False

    result[label]['stat'] = round(stats,2)
    result[label]['p-value'] = p

df_norm2 = pd.DataFrame(result)

print(df_norm2.T.to_string())


# Pearson correlation table
# Pick features of interest

df_corr = df_norm[['stars','review_count','avg_review_age','hours_per_week','est_business_age', 'highly_rated']]

# Calculate pearson correlation
print(df_corr.corr().round(2))

# Sns heatmap
df_corr.drop('highly_rated',axis=1).corr().round(2)

sns.heatmap(df_corr.drop('highly_rated',axis=1).corr().round(2))
plt.title('Heatmap of log scaled numeric features')

sns.pairplot(df_corr, hue="highly_rated")
plt.title('Pair plot of selected log scaled features.')

# Desc
print(df_processed.describe().round(2))

# Pie chart
def my_autopct(pct):
    return ('%.2f %%' % pct) if pct > 7.5 else ''


statecounts = df['state'].value_counts()
statecounts = statecounts[statecounts.values > 1]

plt.pie(statecounts, labels=statecounts.keys(), autopct=my_autopct)

plt.title('Proportion of restaurants coming from each state')

plt.show()


# KDE
kde_df = df_processed.copy()
kde_df = kde_df[days].drop('hours_per_week', axis=1)
result_df = pd.DataFrame(columns=['hours_open', 'Day'])

for column in kde_df.columns:
    temp_df = pd.DataFrame({'hours_open': kde_df[column]})
    temp_df['Day'] = column
    result_df = pd.concat([result_df, temp_df])

result_df = result_df.reset_index()

sns.kdeplot(data=result_df, x='hours_open', hue='Day', multiple='stack')
plt.title('Kernel Density Function of Hours Open during each day of the week')
plt.xlabel('Hours Open')
plt.show()