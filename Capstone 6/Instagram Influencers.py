#!/usr/bin/env python
# coding: utf-8

# # Instagram Influencers

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# # Loading Dataset

# In[3]:


# load the data
df=pd.read_csv("C:\\Users\\Anjali\\Downloads\\Influencer.csv")
df.sample(5)


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# # Data Cleaning

# In[6]:


df['Channel Info'] = df['Channel Info'].str.replace('\n','')
df.sample(5)


# In[18]:


# Convert m,k,b numeric
number_list = ['Followers', 'Avg. Likes', 'Posts', 'New Post Avg. Likes', 'Total Likes']
tbl = {'k': 1000, 'm': 1000000, 'b': 1000000000}
for col in number_list:
    lst = df[col]
    df[col] = [int(re.sub(r'([\d\.]+)(k|m|b)', lambda v: str(int(float(v.groups()[0]) * tbl[v.groups()[1]])), str(val))) for val in lst]


# In[25]:


df[df['Country Or Region']=='United States']


# In[23]:


df.head()


# In[21]:


df.describe().T


# # 1.Are there any correlated features in the given dataset? If yes, state the correlation
# 
# 
# 
# 
# # coefficient of the pair of features which are highly correlated.

# In[22]:


# Calculate correlation matrix
correlation_matrix = df.corr()

# Find highly correlated feature pairs
highly_correlated = correlation_matrix[abs(correlation_matrix) > 0.7].stack().drop_duplicates()

# Print the highly correlated feature pairs and their correlation coefficients
print("Highly correlated features:")
for (feature1, feature2), correlation in highly_correlated.items():
    if feature1 != feature2:
        print(f"{feature1} - {feature2}: {correlation}")

# 2. What is the frequency distribution of the following features?
○ Influence Score
○ Followers
○ Posts
# In[26]:


#  Frequency distribution
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
features = ['Influence Score', 'Followers', 'Posts']
colors = ['lightblue', 'orange', 'pink']

for i, feature in enumerate(features):
    ax = axes[i]
    ax.hist(df[feature], bins=20, color=colors[i], ec='red')
    ax.set_title(f'{feature} Frequency Distribution')
plt.tight_layout()
plt.show()


# # 3. Which country houses the highest number of Instagram Influencers? Please show the
# 
# 
# 
# 
# # count of Instagram influencers in different countries using barchart

# In[27]:


# Count of influencers by country
country_counts = df['Country Or Region'].value_counts()
plt.bar(country_counts.index, country_counts.values, color='violet', ec='blue')
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Count of Instagram Influencers by Country')
plt.xticks(rotation=90)
plt.show()


# # 4. Who are the top 10 influencers in the given dataset based on the following features

# Top 10 Followers

# In[28]:


# Top 10 Followers
dfi = df.loc[:9]
plt.figure(figsize=(8,6))
sns.barplot(data=dfi, y="Channel Info", x="Followers", palette="Spectral")
None


# # Top 10 Average Likes

# In[29]:


# Top 10 Averag Likes
dfx = df.sort_values(by='Avg. Likes' , ascending=False)[0:10]
plt.figure(figsize=(8,6))
sns.barplot(data=dfx, y="Channel Info", x='Avg. Likes', palette="flare")
None


# # Top 10 Total Likes

# In[30]:


#  Top  10 Total Likes
dfx = df.sort_values(by='Total Likes' , ascending=False)[0:10]
plt.figure(figsize=(8,6))
sns.barplot(data=dfx, y="Channel Info", x='Total Likes', palette="magma")
None


# 5. Describe the relationship between the following pairs of features using a suitable graph
# ● Followers and Total Likes
# ● Followers and Influence Score
# ● Posts and Average likes
# ● Posts and Influence Score

# In[31]:


#  Relationship between pairs of features
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
features_pairs = [('Followers', 'Total Likes'),
                 ('Followers', 'Influence Score'),
                 ('Posts', 'Avg. Likes'),
                 ('Posts', 'Influence Score')]
colors = ['cyan', 'purple', 'green', 'magenta'] 

for i, (x, y) in enumerate(features_pairs):
    ax = axes[i // 2, i % 2]
    ax.scatter(df[x], df[y], color=colors[i])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f'{x} vs {y}')
plt.tight_layout()
plt.show()

