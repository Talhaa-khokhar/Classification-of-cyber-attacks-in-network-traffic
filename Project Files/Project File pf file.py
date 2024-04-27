#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# In[2]:


df = pd.read_csv('Dataset.txt')
df.head()


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


attack_type = pd.read_csv('Attack_types.txt')


# In[6]:


attack_type.shape


# In[7]:


df.shape


# In[1]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[10]:


df.describe()


# In[11]:


df.columns


# In[12]:


df.columns = ['Duration', 'Protocol Type', 'Service', 'Flag', 'Source Bytes','Destination Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot',
       'Num Failed Logins', 'LoggedIn', 'Num Compromised', 'Root Shell',
       'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells',
       'Num Access Files', 'Num Outbound cmds', 'Is Host Login',
       'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate',
       'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate',
       'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count',
       'Dst Host Srv Count', 'Dst Host Same Srv Rate',
       'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate',
       'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate',
       'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate',
       'Dst Host Srv Rerror Rate', 'Attack Category', 'Occurance']


# In[13]:


len(df.columns)


# In[14]:


df.columns


# In[15]:


df


# In[16]:


attack_type.head()


# In[17]:


attack_type.shape


# In[1]:


attack_type.columns = ['Attack Category Attack Type']


# In[2]:


attack_type.head()


# In[20]:


df.describe()


# In[21]:


plt.hist(df['Duration'], bins=10)
plt.show()


# In[22]:


plt.hist(df['Num Compromised'], bins=10)
plt.show()


# In[23]:


df.drop_duplicates(inplace=True)


# In[24]:


df


# In[25]:


df.dropna()


# In[26]:


df.shape


# In[27]:


df['Service'].describe()


# In[28]:


df['Service'].value_counts()


# In[29]:


df['Destination Bytes'].describe()


# In[30]:


df['Destination Bytes'].value_counts()


# In[31]:


df['Source Bytes'].describe()


# In[32]:


df['Source Bytes'].value_counts()


# In[33]:


df['Protocol Type'].describe()


# In[34]:


df['Protocol Type'].value_counts()


# In[35]:


df.corr()


# In[36]:


q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
iqr = q3 - q1

# identify outliers
outliers = df[(df < q1 - 1.5*iqr) | (df > q3 + 1.5*iqr)]

# print the outliers
print(outliers)


# In[37]:


df


# In[38]:


sns.distplot(df['Source Bytes'], color = "g")


# In[39]:


sns.distplot(df["Destination Bytes"],bins=10,kde=False)


# In[40]:


sns.lineplot( df['Source Bytes'], df['Destination Bytes'])


# In[41]:


sns.scatterplot(df['Source Bytes'], df['Destination Bytes'])


# In[42]:


sns.scatterplot( df['Source Bytes'], df['Destination Bytes'], hue =df["Flag"])


# In[43]:


sns.boxplot( df['Destination Bytes'] )


# In[44]:


sns.boxplot(  df['Source Bytes'], df['Destination Bytes'])


# In[45]:


sns.violinplot(df['Source Bytes'], df['Destination Bytes'])


# In[46]:


sns.countplot(df["Flag"])


# In[47]:


attack_type


# In[48]:


df.columns


# In[49]:


newdata= df[['Destination Bytes', 'Land', 'Wrong Fragment', 'Urgent', 'Hot',
       'Num Failed Logins', 'LoggedIn', 'Num Compromised', 'Root Shell',
       'Su Attempted', 'Num Root', 'Num File Creations', 'Num Shells',
       'Num Access Files', 'Num Outbound cmds', 'Is Host Login',
       'Is Guest Login', 'Count', 'Srv Count', 'Serror Rate',
       'Srv Serror Rate', 'Rerror Rate', 'Srv Rerror Rate', 'Same Srv Rate',
       'Diff Srv Rate', 'Srv Diff Host Rate', 'Dst Host Count',
       'Dst Host Srv Count', 'Dst Host Same Srv Rate',
       'Dst Host Diff Srv Rate', 'Dst Host Same Src Port Rate',
       'Dst Host Srv Diff Host Rate', 'Dst Host Serror Rate',
       'Dst Host Srv Serror Rate', 'Dst Host Rerror Rate',
       'Dst Host Srv Rerror Rate',  'Occurance']]


# In[50]:


df['Attack Category'].describe()


# In[51]:


df['Attack Category'] == '<'


# In[52]:


corr = df.corr()
plt.figure(figsize=(30,30))
#sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))
#sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns, annot=True,cmap=sns.diverging_palette(220, 20, as_cmap=True))

sns.heatmap(corr, annot=True, cmap="coolwarm", linewidths=0.5, linecolor="gray", fmt='.2f')
plt.show()


# In[53]:


X=newdata


# In[54]:


y=df['Attack Category']


# In[55]:



X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=42)


# In[56]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[57]:


from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression(random_state=1) 
model1.fit(X_train, y_train) 

y_pred1 = model1.predict(X_test) 
print(classification_report(y_test, y_pred1)) 


# In[58]:


accuracy_score(y_test, y_pred1)


# # KNN

# In[66]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# In[67]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(x=['Accuracy', 'Precision', 'Recall', 'F1 Score'], y=[accuracy, precision, recall, f1])


# # DecisionTree

# In[68]:


dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[69]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(x=['Accuracy', 'Precision', 'Recall', 'F1 Score'], y=[accuracy, precision, recall, f1])


# # ANN

# In[70]:


from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50,30), max_iter=1000, random_state=1)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)


# In[71]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')


print('Accuracy:', accuracy)
print('Precision:', precision)
print('Recall:', recall)
print('F1 score:', f1)

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(x=['Accuracy', 'Precision', 'Recall', 'F1 Score'], y=[accuracy, precision, recall, f1])


# # k -means

# In[73]:


# Use elbow method to find optimal number of clusters
from sklearn.cluster import KMeans

SSE = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=1)
    kmeans.fit(X_train)
    SSE.append(kmeans.inertia_)
plt.plot(range(1, 11), SSE)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()


# In[76]:




model2 = KNeighborsClassifier() 
model2.fit(X_train, y_train) 

y_pred2 = model2.predict(X_test)
print(classification_report(y_test, y_pred2)) 


# In[77]:


accuracy_score(y_test, y_pred2)


# In[78]:


from sklearn.svm import SVC

model3 = SVC(random_state=1) 
model3.fit(X_train, y_train) 

y_pred3 = model3.predict(X_test) 
print(classification_report(y_test, y_pred3))


# In[79]:


accuracy_score(y_test, y_pred3)


# In[80]:




model5 = DecisionTreeClassifier(random_state=1) 
model5.fit(X_train, y_train) 

y_pred5 = model5.predict(X_test) 
print(classification_report(y_test, y_pred5)) 


# In[81]:


accuracy_score(y_test, y_pred5)


# In[82]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred2)
print(cm)
accuracy_score(y_test, y_pred2)


# In[ ]:





# In[ ]:




