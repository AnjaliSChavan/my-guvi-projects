#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis

# # Import libraries which are necessary

# In[6]:


import re
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from nltk.stem import WordNetLemmatizer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report


# # Read and Load the Dataset

# In[8]:


#importing the dataset
Dataset_Columns=['target', 'ids', 'date', 'flag', 'user', 'text']
Dataset_Encoding="ISO-8859-1"

df=pd.read_csv("C:\\Users\\Anjali\\Downloads\\twitter_new.csv", encoding=Dataset_Encoding, names=Dataset_Columns)
df.sample(5)


# # Exploratory Data Analysis
# 

# In[9]:


# check five top & bottom records of data
df.head()


# In[10]:


df.tail()


# In[11]:


# columns/features in data
df.columns


# In[12]:


# length of the dataset
print("length of data is", len(df))


# In[13]:


# shape of data
df.shape


# In[14]:


# data information
df.info()


# In[15]:


# data types of all columns
df.dtypes


# In[16]:


# checking for null values
np.sum(df.isnull().any(axis=1))


# In[17]:


# Rows & Columns in data set
print("No of Columns in the data is:", len(df.columns))
print("No of Rows in the data is:", len(df))


# In[18]:


# check unique target values
df['target'].unique()


# In[19]:


# check the no. of target values
df['target'].nunique()


# # Data Visualization of Target Variables
# 

# In[20]:


# plotting the distribution for dataset
ax=df.groupby('target').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative', 'Positive'],rotation=0)

# storing data in lists
text, sentiment = list(df['text']), list(df['target'])


# In[21]:


# using seaborn
sns.countplot(x='target', data=df)


# # Data Preprocessing

# In[22]:


# Selecting the text and target column for further analysis
data=df[['text', 'target']]


# In[23]:


# Replacing the values to ease understanding
# Assigning 1 to positive sentiment from 4
data['target']=data['target'].replace(4,1)


# In[24]:


# Printing unique value of target variables
data['target'].unique()


# In[25]:


# Separating positive and negative tweets
data_pos=data[data['target'] == 1]
data_neg=data[data['target'] == 0]


# In[26]:


# Taking 1/4th of the data so we can run it on our machine easily
data_pos=data_pos.iloc[:int(20000)]
data_neg=data_neg.iloc[:int(20000)]


# In[27]:


# Combining positive and negative tweets
dataset=pd.concat([data_pos, data_neg])


# In[28]:


# Making statement in lowercase
dataset['text']=dataset['text'].str.lower()
dataset['text'].tail()


# In[30]:


# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad', 
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed', 
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink', 
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}


# In[31]:


# Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from', 
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']


# In[32]:


def preprocess(textdata):
    processedText = []
    
    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()
    
    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"
    
    for tweet in textdata:
        tweet = tweet.lower()
        
        # Replace all URls with 'URL'
        tweet = re.sub(urlPattern,' URL',tweet)
        # Replace all emojis.
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])        
        # Replace @USERNAME to 'USER'.
        tweet = re.sub(userPattern,' USER', tweet)        
        # Replace all non alphabets.
        tweet = re.sub(alphaPattern, " ", tweet)
        # Replace 3 or more consecutive letters by 2 letter.
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            # Checking if the word is a stopword.
            #if word not in stopwordlist:
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')
            
        processedText.append(tweetwords)

    return processedText


# In[33]:


# Separating input feature and label
X=data.text
y=data.target


# In[34]:


# Word cloud for Negative word
data_neg = data['text'][:800000]
plt.figure(figsize = (20,20))
Negative_WC = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(Negative_WC)


# In[35]:


# Word cloud for Positive word
data_pos = data['text'][800000:]
Positive_WC = WordCloud(max_words = 1000 , width = 1600 , height = 800,
              collocations=False).generate(" ".join(data_pos))
plt.figure(figsize = (20,20))
plt.imshow(Positive_WC)


# # Splitting the data into train and test

# In[36]:


# Separating the 95% data for training and 5% for testing data
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.05, random_state=0)


# # Feature Extraction

# # Transforming the dataset using TF-IDF Vectorizer

# In[37]:


# Fit the TF-IDF Vectorizer
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names_out()))


# In[38]:


# Transform the data using TF-IDF Vectorizer
X_train=vectoriser.transform(X_train)
X_test=vectoriser.transform(X_test)


# # Function for Model Evaluation

# In[39]:


def model_Evaluate(model):
    # Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)


# # Model Building

# # Naive Bayes model -1

# In[40]:


# Model-1 Naive_Bayes - Bernoulli
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)


# In[41]:


# Plot ROC-AUC Curve for Naive Bayes
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred1)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='violet', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# # Support Vector Classifier Model-2

# In[42]:


# Model-2 SVC 
SVCmodel = LinearSVC()
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)


# In[43]:


# Plot ROC-AUC Curve for SVC
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred2)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# # Logistic Regression Model-3
# 

# In[44]:


# Model-3 Logistic Regression
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)


# In[45]:


#Plot the ROC-AUC Curve for LR
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred3)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE')
plt.legend(loc="lower right")
plt.show()


# # Saving and Train the model
# 

# In[46]:


# saving the model using joblib
from joblib import dump
dump(vectoriser,'vectoriser_train.joblib')
dump(BNBmodel,'Naive_bayes_model1.joblib')
dump(SVCmodel,'SVC_model2.joblib')
dump(LRmodel,'LR_model3.joblib')


# In[47]:


# Train the model
from joblib import load

def load_models():
    #load the vectoriser
    vectoriser=load('vectoriser_train.joblib')
    BNBmodel=load('Naive_bayes_model1.joblib')
    SVCmodel=load('SVC_model2.joblib')
    LRmodel=load('LR_model3.joblib')
    return vectoriser,BNBmodel,SVCmodel,LRmodel

def predict(vectoriser, model, text):
    # Predict the sentiment
    textdata = vectoriser.transform(preprocess(text))
    sentiment = model.predict(textdata)
    
    # Make a list of text with sentiment
    data = [(t, s) for t, s in zip(text, sentiment)]
    
    # Convert the list into a Pandas DataFrame
    df = pd.DataFrame(data, columns=['text', 'sentiment'])
    df['sentiment'] = df['sentiment'].replace([0, 1], ['Negative', 'Positive'])  # Replace sentiment labels
    return df

    
if __name__=="__main__":
    
    # Loading the models and vectorizer
    vectoriser,BNBmodel,SVCmodel,LRmodel=load_models()
    
    # Text to classify should be in a list
    text = ["I hate twitter",
            "May the Force be with you.",
            "Mr. Stark, I don't feel so good"]
    
     # Predict using Bernoulli Naive Bayes model
    df_bnb = predict(vectoriser, BNBmodel, text)
    print("Bernoulli Naive Bayes Model:")
    print(df_bnb)

    # Predict using Support Vector Classifier model
    df_svc = predict(vectoriser, SVCmodel, text)
    print("\nSupport Vector Classifier Model:")
    print(df_svc)

    # Predict using Logistic Regression model
    df_lr = predict(vectoriser, LRmodel, text)
    print("\nLogistic Regression Model:")
    print(df_lr)


# In[ ]:




