#!/usr/bin/env python
# coding: utf-8

# # Guvi Rating Prediction

# # Importing Libraries And Loading Dataset
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px 
import seaborn as sns
from scipy import stats
import warnings 
warnings.filterwarnings('ignore')


# # Loading the Dataset

# In[2]:


df = pd.read_csv("C:\\Users\\Anjali\\Downloads\\3.1-data-sheet-guvi-courses.csv") 
df


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


# Categorical columns
cat_columns=list(df.select_dtypes(['object']).columns)
print("Categorical Columns")
print(cat_columns)


# In[6]:


# Numerical Columns
Num_columns=list(df.select_dtypes(['int','float']))
print('Numerical Columns')
print(Num_columns)


# In[7]:


df.describe()


# In[8]:


#fig, axes = plt.subplots(4,2,figsize=(25,20))
#i=0
#j=0
for k in range(len(Num_columns)):
  col=Num_columns[k]
  #if j == 1:
    #j=0
   # i=i+1
  #sns.displot(df[col],kde=True,ax=axes[i,j])
  sns.displot(df[col],kde=True, color='purple')
  plt.legend(labels=["Skewness: %.2f"%(df[col].skew())])
 


# In[9]:


df.isnull().sum()


# In[10]:


df.dropna(inplace=True)


# In[11]:


df.isnull().sum()


# In[12]:


df.info()


# Out of 3680 rows totaly 4 rows are eliminated; now there are 3676 rows in df after removing null values

# # Remove columns which are unwanted

# In[13]:


df


# In[14]:


df.nunique()


# In[15]:


df['course_id'].value_counts()


# In[16]:


df['course_title'].value_counts()


# In[17]:


df['url'].value_counts()


# In[18]:


df['published_timestamp'].value_counts()


# In[19]:


df['Rating'].value_counts()


# If the features 'course_id', 'course_title', 'url', and 'published_timestamp' are not related to the dependent variable 'Ratings' and do not provide any meaningful information for predicting ratings, it is reasonable to drop these features from the dataset.

# In[20]:


df.drop(['course_id','course_title','url','published_timestamp'],axis=1,inplace=True)


# In[21]:


df


# # Encoding

# In[22]:


df.info()


# In[23]:


df['level'].value_counts()


# In[24]:


cleanup_level = {"level": {"All Levels": 1, "Beginner Level": 2, "Intermediate Level": 3, "Expert Level": 4}}
df['level'] = df['level'].map(cleanup_level['level'])


# In[25]:


df['subject'].value_counts()


# In[26]:


subject_num = {'subject': {'Subject: Web Development': 1, 'Business Finance': 2, 'Musical Instruments': 3, 'Graphic Design': 4}}
df['subject'] = df['subject'].map(subject_num['subject'])


# In[27]:


df.info()


# In[28]:


#HEATMAP FOR CORRELATION BETWEEN VARIABLES
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[29]:


# Dropping multi collinearity columns 
df.drop(['content_duration','num_reviews'],axis=1,inplace= True)


# In[30]:


df.info()


# In[31]:


# Handling skewed Data
fig,axes=plt.subplots(3,2,figsize=(20,13))
i=0
j=0
for col in list(df.columns):
  if j==2:
    i=i+1
    j=0
  t=sns.histplot(df[col],ax=axes[i,j],kde=True, color='orange')
  t.legend(labels=["Skewness: %.2f"%(df[col].skew())])
  j=j+1


# In[32]:


df.describe()


# In[33]:


fig,axes=plt.subplots(1,2,figsize=(12,8))
i=0
for col in list(df.columns):
    if  any(x<0 for x in list(df[col]))!=True:
        print(f'{col} has Only Positive Values')
        if df[col].skew() >= 10.0:
            t=sns.histplot(df[col],ax=axes[i],kde=True, color='brown')
            t.legend(labels=["Skewness: %.2f"%(df[col].skew())],title='Before Transformation')

           
            new=f'Transformed_{col}'
            df[new]=df[col].apply(lambda i: np.log(i+0.1))
            t1=sns.histplot(df[new],ax=axes[i+1],kde=True, color='green')
            t1.legend(labels=["Skewness: %.2f"%(df[new].skew())],title='After Transformation')
            i=i+1


# In[34]:


df.drop(['num_subscribers'],axis=1,inplace=True)


# # Outliers

# In[35]:


# As these columns are encoded from categorical to ordinal and 'Rating' is Target variable
new_df=df.drop(['subject','level','Rating'],axis=1)


# In[36]:


columns= list(new_df.columns)
columns


# In[37]:


fig, axes = plt.subplots(3, figsize=(15, 10))
colors = ['red', 'green', 'blue']  # Specify the colors for each boxplot
i = 0
for col in columns:
    sns.boxplot(x=col, data=df, orient='h', ax=axes[i], color=colors[i])  # Set the color for each boxplot
    i += 1


# In[38]:


sns.boxplot(df,orient='h')


# In[39]:


plt.figure(figsize=(15,10))
sns.boxplot(x='num_lectures',y='price',data=df,orient='h')
plt.show()


# In[40]:


outlier_index=[]
for i in columns:
    q1=np.percentile(df[i],25,interpolation="midpoint")
    q3=np.percentile(df[i],75,interpolation="midpoint")
    print("q1 value of {i} is {q1}".format(i=i,q1=q1))
    iqr1=q3-q1
    print("q3 value of {i} is {q2}".format(i=i,q2=q3))
    print("iqr value of {i} is {iqr}".format(i=i,iqr=iqr1))
    lower=df.index[df[i]<(q1-1.5*iqr1)]
    lower_limit=q1-1.5*iqr1
    print(f'Lower limit/bound of {i} is {lower_limit}')
    lowercount=np.size(lower)
    print("Lower Outlier of {column} are {lower} counts".format(column=i,lower=lowercount))
    outlier_index.extend(lower)
    upper=df.index[df[i]>(q3+1.5*iqr1)]
    upper_limit=q3+1.5*iqr1
    print(f'Upper limit/bound of {i} is {upper_limit}')
    uppercount=np.size(upper)
    print("Upper Outlier of {column} are {upper} counts".format(column=i,upper=uppercount))
    outlier_index.extend(upper)
    totalcount=lowercount+uppercount
    print("Total Outliers of {column} is {x}".format(column=i,x=totalcount))


# In[41]:


outlier_index.sort()
outlier_index1=sorted(set(outlier_index),key=outlier_index.index)  
print("Total outliers of all columns are",len(outlier_index))
print("Total unique outliers of all columns are",len(outlier_index1))


# In[42]:


# Dropping outliers
df=df.drop(outlier_index1)


# In[43]:


df


# In[44]:


plt.figure(figsize=(15,10))
sns.boxplot(df,orient='h')


# # Scaling Data

# In[45]:


df.info()


# In[46]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
for col in df.columns:
    df[col]=scaler.fit_transform(df[[col]])


# In[47]:


df


# In[48]:


df.info()


# In[49]:


sns.heatmap(df.corr(),annot=True)


# # Model Building and Training 

# # Splitting Datasets as Training and Testing Datasets

# In[50]:


from sklearn.model_selection import train_test_split
y=df['Rating']
x=df.drop(['Rating'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)


# # Linear Regression

# In[51]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
model=LinearRegression()
model.fit(x_train,y_train)
print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))


# In[52]:


# Testing Test
r2 = model.score(x_test,y_test)
print('The r2 score of Testing Dataset for a linear model is', r2)


# # Polynomial Regression Model

# In[53]:


#Fitting the polynomial Regression into data set
from sklearn.preprocessing import PolynomialFeatures

# Transform independent variables to polynomial features
poly = PolynomialFeatures(degree=3)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)


# In[54]:


# Train the model
model = LinearRegression()
model.fit(x_train_poly, y_train)
print("The r2 score of Training dataset for linear model is",model.score(x_train_poly, y_train))


# In[55]:


# Testing Test
r2 = model.score(x_test_poly,y_test)
print('The r2 score of Testing Dataset for a linear model is', r2)


# # Finding Optimal Depth of Tree

# In[56]:


# import the regressor
from sklearn.tree import DecisionTreeRegressor 
train_accuracy=[]
test_accuracy=[]
  
for i in range(1,25):
    # create a regressor object
    model3 = DecisionTreeRegressor(random_state = 1,max_depth=i) 
    
    # fit the regressor with X and Y data
    model3.fit(x_train, y_train)
    train_accuracy.append(model3.score(x_train,y_train))
    print("The r2 score of Training dataset for linear model is:",model3.score(x_train,y_train))
    
    # Testing Test
    test_accuracy.append(model3.score(x_test,y_test))
    print('The r2 score of Testing Dataset for a linear model is:', r2)


# In[57]:


plt.figure(figsize=(5,10))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label='Training_Accuracy')
plt.plot(test_accuracy,label='Testing_Accuracy')
plt.xticks(range(1,26,5))
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy of Model')
plt.legend(loc='upper left')
plt.show()


# Decision Tree model gives max 35% accuracy while testing. So It is better to look for other algorithms

# # Random Forest Model

# In[58]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error as MSE

train_accuracy=[]
test_accuracy=[]

for i in range(1,26):
    # Train the model
    model = RandomForestRegressor(n_estimators=100,max_depth=i, random_state=1)
    model.fit(x_train, y_train)

    # fit the regressor with X and Y data
    model.fit(x_train, y_train)
    train_accuracy.append(model.score(x_train,y_train))
    print("The r2 score of Training dataset for linear model is:",model.score(x_train,y_train))
    
    # Testing Test
    test_accuracy.append(model.score(x_test,y_test))
    print('The r2 score of Testing Dataset for a linear model is:', r2)


# In[59]:


# plotting graph for finding optimal max_depth
plt.figure(figsize=(5,10))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label='Training_Accuracy')
plt.plot(test_accuracy,label='Testing_Accuracy')
plt.xticks(range(0,26,5))
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy of Model')
plt.legend(loc='upper left')
plt.show()


# # Building Random Forest Model with optimum Depth

# 1.Grid Search CV

# In[60]:


rfc=RandomForestRegressor(random_state=42)


# In[61]:


param_grid = { 
    'n_estimators': [200,300,400, 500,1000],
    'max_features': ['sqrt', 'log2'], # Auto is default value
    'max_depth' : [4,5,6,7,8,9,10,11,12,13,14,15]
}


# In[62]:


from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(x_train, y_train)


# In[63]:


CV_rfc.best_params_


# In[64]:


results=pd.DataFrame(CV_rfc.cv_results_)


# In[65]:


results


# In[66]:


CV_rfc.score


# In[67]:


CV_rfc.best_score_


# In[68]:


print("The r2 score of Training dataset for linear model is",CV_rfc.score(x_train,y_train))
# Testing Test
print('The r2 score of Testing Dataset for a linear model is', CV_rfc.score(x_test,y_test))

y_pred=CV_rfc.predict(x_test)
print('The RMSE score of model is ',MSE(y_test,y_pred,squared=False))


# # 2.RandomizedSearchCV

# In[69]:


from sklearn.model_selection import RandomizedSearchCV
CV_rfc1= RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, cv= 5)
CV_rfc1.fit(x_train, y_train)

results1=pd.DataFrame(CV_rfc1.cv_results_)


# In[70]:


results1


# In[71]:


print("The r2 score of Training dataset for linear model is",CV_rfc1.score(x_train,y_train))
# Testing Test
print('The r2 score of Testing Dataset for a linear model is', CV_rfc1.score(x_test,y_test))

y_pred=CV_rfc1.predict(x_test)
print('The RMSE score of model is ',MSE(y_test,y_pred,squared=False))


# # 3.Manual Optimum

# In[72]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error as MSE

# Train the model
model = RandomForestRegressor(n_estimators=500,max_depth=8, random_state=1)
model.fit(x_train, y_train)

# fit the regressor with X and Y data
model.fit(x_train, y_train)
print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))
# Testing Test
print('The r2 score of Testing Dataset for a linear model is', model.score(x_test,y_test))

y_pred=model.predict(x_test)
print('The RMSE score of model is ',MSE(y_test,y_pred,squared=False))


# # GradientBoost Regressor:
# 

# In[73]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error as MSE

train_accuracy=[]
test_accuracy=[]

for i in range(1,26):
    # Train the model
    model = GradientBoostingRegressor(n_estimators=100,max_depth=i, random_state=1)
    model.fit(x_train, y_train)

    # fit the regressor with X and Y data
    model.fit(x_train, y_train)
    train_accuracy.append(model.score(x_train,y_train))
    #print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))
    # Testing Test
    test_accuracy.append(model.score(x_test,y_test))
    #print('The r2 score of Testing Dataset for a linear model is', r2)
    
# plotting graph for finding optimal max_depth
plt.figure(figsize=(5,10))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label='Training_Accuracy')
plt.plot(test_accuracy,label='Testing_Accuracy')
plt.xticks(range(1,26,5))
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy of Model')
plt.legend(loc='upper left')
plt.show()


# # Building GradientBoosting Model with Optimal Depth
# 

# In[74]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error as MSE

train_accuracy=[]
test_accuracy=[]

for i in range(1,26):
    # Train the model
    model = GradientBoostingRegressor(n_estimators=100,max_depth=i, random_state=1)
    model.fit(x_train, y_train)

    # fit the regressor with X and Y data
    model.fit(x_train, y_train)
    train_accuracy.append(model.score(x_train,y_train))
    #print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))
    # Testing Test
    test_accuracy.append(model.score(x_test,y_test))
    #print('The r2 score of Testing Dataset for a linear model is', r2)
    
# plotting graph for finding optimal max_depth
plt.figure(figsize=(5,10))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label='Training_Accuracy')
plt.plot(test_accuracy,label='Testing_Accuracy')
plt.xticks(range(1,26,5))
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy of Model')
plt.legend(loc='upper left')
plt.show()


# # XG Boosting Model

# In[76]:


pip install xgboost


# In[77]:


from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error as MSE

train_accuracy=[]
test_accuracy=[]

for i in range(1,26):
    # Train the model
    model = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2,max_depth=i)
    model.fit(x_train, y_train)

    # fit the regressor with X and Y data
    model.fit(x_train, y_train)
    train_accuracy.append(model.score(x_train,y_train))
    #print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))
    # Testing Test
    test_accuracy.append(model.score(x_test,y_test))
    #print('The r2 score of Testing Dataset for a linear model is', r2)
    
# plotting graph for finding optimal max_depth
plt.figure(figsize=(5,10))
sns.set_style('whitegrid')
plt.plot(train_accuracy,label='Training_Accuracy')
plt.plot(test_accuracy,label='Testing_Accuracy')
plt.xticks(range(0,26,5))
plt.xlabel('Depth of Tree')
plt.ylabel('Accuracy of Model')
plt.legend(loc='upper left')
plt.show()


# In[78]:


from xgboost import XGBRFRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Train the model
model = XGBRFRegressor(n_estimators=100, subsample=0.9, colsample_bynode=0.2,max_depth=15)
model.fit(x_train, y_train)

# fit the regressor with X and Y data
model.fit(x_train, y_train)
print("The r2 score of Training dataset for linear model is",model.score(x_train,y_train))
# Testing Test
r2 = model.score(x_test,y_test)
print('The r2 score of Testing Dataset for a linear model is', r2)
y_pred=model.predict(x_test)
print('The RMSE score of model is ',MSE(y_test,y_pred,squared=False))


# # Result

# From Trial and testing different models, Random Forest model gives comparatively somewhat better accuracy i.e 40%. So Finally Random Forest Regressor model is selected even though there is lot of scope to work on this project.

# In[ ]:





# In[ ]:





# In[ ]:




