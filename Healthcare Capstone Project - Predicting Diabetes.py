#!/usr/bin/env python
# coding: utf-8

# # Healthcare Capstone Project - Predicting Diabetes
# 
# **<u>DESCRIPTION<u>**
# 
# - The dataset used in this project is from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
# - The objective of this project is to predict whether or not a patient has diabetes based on certain diagnostic measurements in the dataset.
# - Diabetes is a chronic, costly, and consequential disease, making accurate predictions critical for patient health and well-being.
# - The model built in this project aims to accurately predict diabetes status for patients in the dataset, potentially providing valuable insights for medical professionals.
# - The success of this project could lead to the development of more effective diabetes diagnostic tools and treatments, improving patient outcomes and reducing the burden of diabetes on individuals and healthcare systems.

# # Week 1: Data Exploration:
# 
# 1. Perform descriptive analysis. Understand the variables and their corresponding values. On the columns below, a value of zero does not make sense and thus indicates missing value:
# 	- Glucose   
# 	- BloodPressure 
# 	- SkinThickness
# 	- Insulin
# 	- BMI
#     
# 
# 2. Visually explore these variables using histograms. Treat the missing values accordingly.
# 
# 3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables.

# **<u>Read Data<u>**

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import accuracy_score,classification_report,roc_curve,auc


# In[2]:


df= pd.read_csv("health care diabetes.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# **<u>Descriptive Analysis<u>**

# In[5]:


df.isnull().any()


# In[6]:


df.dtypes


# In[7]:


df.describe()


# In[8]:


df.info()


# **<u>Observation from Descriptive Analysis<u>**
# - The mean glucose level of the patients is 120.89 and the mean BMI is 31.99.
# - On average, patients have undergone 3.84 pregnancies.
# - The minimum glucose level, blood pressure, and skin thickness values are 0, which indicates that there are some missing or - erroneous values in the dataset.
# - The maximum number of pregnancies for a single patient is 17, and the maximum BMI value is 67.1.
# - The target variable is Outcome, which is binary with 0 indicating no diabetes and 1 indicating diabetes. The mean value of the - outcome variable is 0.35, which suggests that approximately 35% of patients in the dataset have diabetes.
# - The standard deviation of insulin is quite large at 115.24, indicating a significant amount of variability in this feature.
# - The age of the patients ranges from 21 to 81 years, with a mean of 33.24.

# **<u>2. Visually explore these variables using histograms. Treat the missing values accordingly.<u>**

# - <u>**GLUCOSE**<u>

# In[9]:


sns.set_style("whitegrid")
sns.displot(data=df,x="Glucose",kde=True)
plt.xlabel("Glucose Level")
plt.title("Glucose Chart")


# Here most the values are between 100-125 

# In[10]:


df=df.replace({"Glucose":{0:np.nan}})
mean=df["Glucose"].mean()
df["Glucose"].fillna(mean,inplace=True)


# In[11]:


sns.set_style("darkgrid")
sns.displot(data=df,x="Glucose",kde=True)
plt.xlabel("Glucose Level")
plt.title("Glucose After Fixing Null values")


# - **<u>BLOOD PRESSURE<u>**

# In[12]:


sns.set_style("whitegrid")
sns.displot(data=df,x="BloodPressure",kde=True,color="r")
plt.title("BloodPressure Chart")
plt.xlabel("BloodPressure level")


# Here most of the data are between 60 and 80

# In[13]:


df=df.replace({"BloodPressure":{0:np.nan}})
mean=df["BloodPressure"].mean()
df["BloodPressure"].replace(mean,inplace=True)


# In[14]:


df["BloodPressure"]=df["BloodPressure"].replace(0,df["BloodPressure"].mean())


# In[15]:


sns.set_style("darkgrid")
sns.displot(df["BloodPressure"],kde=True,color="r")
plt.title("BloodPressure Chart")
plt.xlabel("BloodPressure level")


# - <u>**SkinThickness**<u>

# In[16]:


sns.set_style("whitegrid")
sns.displot(df["SkinThickness"],kde=True,color="green")
plt.title("SkinThickness Chart")
plt.xlabel("SkinThickness Level")


# Most the values between 20 and 40. 0 wil be the null values

# In[17]:


df=df.replace({"SkinThickness":{0:np.nan}})
mean=df["SkinThickness"].mean()
df["SkinThickness"].fillna(mean,inplace=True)


# In[18]:


sns.set_style("darkgrid")
sns.displot(df["SkinThickness"],kde=True,color="green",bins=10)
plt.title("SkinThickness Chart After removing Null Values")
plt.title("SkinThickness Level")


# - <u>**INSULIN**<u>

# In[19]:


sns.set_style("whitegrid")
sns.displot(df["Insulin"],kde=True,color="pink")
plt.title("Insulin Chart")
plt.xlabel("Insulin level")


# In[20]:


df=df.replace({"Insulin":{0:np.nan}})
mean=df["Insulin"].mean()
df["Insulin"].fillna(mean,inplace=True)


# In[21]:


sns.set_style("darkgrid")
sns.displot(df["Insulin"],kde=True,color="pink",bins=10)
plt.title("Insulin Chart after removing Zeros from it")
plt.xlabel("Insulin Level")


# - <u>**BMI**<u>

# In[22]:


sns.set_style("whitegrid")
sns.displot(df["BMI"],kde=True,color="blue")
plt.title("BMI Chart")
plt.xlabel("BMI Level")


# In[23]:


df=df.replace({"BMI":{0:np.nan}})
mean=df["BMI"].mean()
df["BMI"].fillna(mean,inplace=True)


# In[24]:


sns.set_style("whitegrid")
sns.displot(df["BMI"],kde=True,color="blue")
plt.title("BMI Chart after removing zeros")
plt.xlabel("BMI Level")


# In[25]:


plt.figure(figsize=(6,5),dpi=100)
sns.heatmap(df.isnull(),cmap="magma",yticklabels=False)


# In[26]:


df["BloodPressure"].describe()


# In[27]:


df["BloodPressure"].isnull()


# In[28]:


df["BloodPressure"].isnull().sum()


# In[29]:


df[df.isnull().any(axis=1)]


# There are 35 Null values, Replacing that with mean

# In[30]:


mean_value = df['BloodPressure'].mean()
df['BloodPressure'].fillna(mean_value, inplace=True)


# In[31]:


plt.figure(figsize=(6,5),dpi=100)
sns.heatmap(df.isnull(),cmap="magma",yticklabels=False)
plt.title("df After Removing Null Values")


# In[32]:


for i in df.columns:
    print("Column Names",i,"has total values of",len(df[i].unique()))
    print(df[i].unique)
    print("-"*100)


# <u>**3. There are integer and float data type variables in this dataset. Create a count (frequency) plot describing the data types and the count of variables**.<u>

# In[33]:


df.dtypes


# In[34]:


df.dtypes.value_counts().plot(kind="bar")


# # Project Task: Week 2
# 
# **<u>Data Exploration:<u>**
# 
# 1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action.
# 
# 2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.
# 
# 3. Perform correlation analysis. Visually explore it using a heat map.

# **<u>1. Check the balance of the data by plotting the count of outcomes by their value. Describe your findings and plan future course of action<u>**

# In[35]:


df["Outcome"].value_counts()


# In[36]:


plt.figure(figsize=(5,4),dpi=100)
sns.countplot(data=df,x="Outcome")
plt.xticks([0,1],["False","True"])


# **<u>2. Create scatter charts between the pair of variables to understand the relationships. Describe your findings.**<u>

# In[37]:


plt.figure(figsize=(6,5))
sns.countplot(data=df,x="Outcome")
plt.xticks([0,1],["False","True"])


# Since We can see there is a imbalance in dataset. We can treat this imbalance with oversampling method.

# In[38]:


df_X=df.drop("Outcome",axis=1)
df_y=df["Outcome"]
print(df_X.shape,df_y.shape)


# In[39]:


pip install imblearn


# In[40]:


from imblearn.over_sampling import SMOTE


# In[41]:


df_X_resample,df_y_resample=SMOTE(random_state=1).fit_resample(df_X,df_y)
print(df_X_resample.shape,df_y_resample.shape)


# In[42]:


plt.figure(dpi=80)
sns.countplot(data=df,x=df_y_resample)
plt.xticks([0,1],["False","True"])
plt.xlabel("Outcome")
plt.ylabel("Count")
df_y_resample.value_counts()


# 2. **<u>Create scatter charts between the pair of variables to understand the relationships. Describe your findings.**<u>

# In[43]:


df_resample=pd.concat([df_X_resample,df_y_resample],axis=1)
df_resample.head()


# In[44]:


df_resample.tail()


# In[45]:


sns.pairplot(data=df_resample,palette="turbo_r",kind="reg")


# In[46]:


sns.pairplot(data=df_resample,palette="turbo",hue="Outcome")


# In[47]:


sns.set_style("white")
sns.jointplot(data=df_resample,x="BloodPressure",y="SkinThickness",hue="Outcome")


# We have some interesting observations from above scatter plot of pairs of features:
# 
# - Glucose alone is impressively good to distinguish between the Outcome classes.
# - Age alone is also able to distinguish between classes to some extent.
# - It seems none of pairs in the dataset is able to clealry distinguish between the Outcome classes.
# - We need to use combination of features to build model for prediction of classes in Outcome.

# **<u> 3. Perform correlation analysis. Visually explore it using a heat map:**<u>

# In[48]:


df_resample.corr()


# In[49]:


plt.figure(dpi=100)
sns.heatmap(df_resample.corr(),cmap="bwr",annot=True)


# Observation:
# 
# - In this there is no negative correlation
# - If are you high positive correlated feature those are;
# 	- Age-Pregnacies
#     - Outcome- Glucose
#     - BMI-Skinthickness

# # Week 3:
# 
# **<u>Data Modeling:**<u>
# 
# **<u>1. Devise strategies for model building. It is important to decide the right validation framework. Express your thought process:**<u>

# Since this is a classification problem, we will be building all popular classification models for our training data and then compare performance of each model on test data to accurately predict target variable (Outcome):
# 
# - Logistic Regression
# - Decision Tree
# - Random Forest
# - Support Vector Machine (SVM)
# - K-Nearest Neighbors (KNN)
# - Naive Bayes
# 
# We will use use GridSearchCV with Cross Validation (CV) = 5 for training and testing model which will give us insight about model performance on versatile data. It helps to loop through predefined hyperparameters and fit model on training set. GridSearchCV performs hyper parameter tuning which will give us optimal hyper parameters for each of the model. We will again train model with these optimized hyper parameters and then predict test data to get metrics for comparing all models.

# In[50]:


from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, confusion_matrix, classification_report, auc, roc_curve, roc_auc_score, precision_recall_curve


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(df_X_resample, df_y_resample, test_size=0.15, random_state =10)


# In[52]:


X_train.shape, X_test.shape


# **<u>2. Apply an appropriate classification algorithm to build a model. Compare various models with the results from KNN algorithm**<u>

# In[206]:


model = []
modelaccuracy = []
modelf1 = []
modelauc = []


# - - KNN 

# In[54]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)


# In[55]:


knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)


# In[59]:


accuracy_knn=accuracy_score(y_test,knn_pred)
f1_knn=f1_score(y_test,knn_pred)


# In[60]:


print("Model Validation\n")
print("\nAccuracy Score\n",accuracy_knn)
print("\nF1 Score\n",f1_knn)
print("\nClassfication report\n",classification_report(y_test,knn_pred))


# In[61]:


knn.score(X_train,y_train)


# In[62]:


knn.score(X_test,y_test)


# Performance evaluation and optimizing parameters using GridSearchCV:

# In[63]:


knn_neighbors = [i for i in range(2,16)]
parameters = {
    'n_neighbors': knn_neighbors
}


# In[64]:


from sklearn.model_selection import GridSearchCV


# In[65]:


gs_knn = GridSearchCV(estimator=knn, param_grid=parameters, cv=5, verbose=0)
gs_knn.fit(df_X_resample, df_y_resample)


# In[66]:


gs_knn.best_params_


# In[67]:


gs_knn.best_score_


# In[68]:


# gs_knn.cv_results_
gs_knn.cv_results_['mean_test_score']


# In[69]:


knn2 = KNeighborsClassifier(n_neighbors=3)


# In[70]:


knn2.fit(X_train, y_train)


# In[71]:


knn2.score(X_train,y_train)


# In[72]:


knn2.score(X_test,y_test)


# In[73]:


# Preparing ROC Curve (Receiver Operating Characteristics Curve)

probs = knn2.predict_proba(X_test)               # predict probabilities
probs = probs[:, 1]                              # keep probabilities for the positive outcome only

auc_knn = roc_auc_score(y_test, probs)           # calculate AUC
print('AUC: %.3f' %auc_knn)
fpr, tpr, thresholds = roc_curve(y_test, probs)# calculate roc curve
plt.figure(figsize=(6,5),dpi=80)
plt.plot([0, 1], [0, 1], linestyle='--')         # plot no skill
plt.plot(fpr, tpr, marker='.')                   # plot the roc curve for the model
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.grid(True,linestyle="-.",color="grey",alpha=0.5,linewidth=0.5)
plt.title("ROC (Receiver Operating Characteristics) Curve");


# In[74]:


# Precision Recall Curve 

pred_y_test = knn2.predict(X_test)                                     # predict class values
precision, recall, thresholds = precision_recall_curve(y_test, probs) # calculate precision-recall curve
f1 = f1_score(y_test, pred_y_test)                                    # calculate F1 score
auc_knn_pr = auc(recall, precision)                                    # calculate precision-recall AUC
ap = average_precision_score(y_test, probs)                           # calculate average precision score
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_knn_pr, ap))
plt.figure(figsize=(6,5),dpi=80)
plt.plot([0, 1], [0.5, 0.5], linestyle='--')                          # plot no skill
plt.plot(recall, precision, marker='.')                               # plot the precision-recall curve for the model
plt.grid(True,linestyle="-.",color="grey",alpha=0.5,linewidth=0.5)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve");


# In[207]:


model.append('KNN')
modelaccuracy.append(accuracy_score(y_test, pred_y_test))
modelf1.append(f1)
modelauc.append(auc_knn)


# - - Logistic Regression

# In[76]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=300)


# In[77]:


lr.fit(X_train,y_train)


# In[78]:


lr_pred=lr.predict(X_test)


# In[79]:


lr.score(X_train,y_train)


# In[80]:


lr.score(X_test,y_test)


# Performance evaluation and optimizing parameters using GridSearchCV: Logistic regression does not really have any critical hyperparameters to tune. However we will try to optimize one of its parameters 'C' with the help of GridSearchCV. So we have set this parameter as a list of values form which GridSearchCV will select the best value of parameter.

# In[81]:


from sklearn.model_selection import GridSearchCV, cross_val_score


# In[82]:


parameters = {'C':np.logspace(-5, 5, 50)}


# In[83]:


gs_lr = GridSearchCV(lr, param_grid = parameters, cv=5, verbose=0)
gs_lr.fit(df_X_resample, df_y_resample)


# In[84]:


gs_lr.best_params_


# In[85]:


gs_lr.best_score_


# In[86]:


lr2=LogisticRegression(C=0.30888435964774846,max_iter=300)
lr2.fit(X_train,y_train)
lr2_pred=lr2.predict(X_test)


# In[87]:


lr2.score(X_train,y_train)


# In[88]:


lr2.score(X_test,y_test)


# In[89]:


#Checking the Accuracy Score
accuracy_lr=accuracy_score(y_test,lr2_pred)


# In[90]:


accuracy_lr


# In[91]:


#AUC ROC Curve
probs=lr2.predict_proba(X_test)
probs_lr=probs[:,1]

fpr,tpr,threshold=roc_curve(y_test,probs_lr)
roc_auc_lr=auc(fpr,tpr)
plt.figure(figsize=(6,5),dpi=80)
plt.title("ROC CUrve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot(fpr,tpr,color="r",label="AUC Score=%0.2f"%roc_auc_lr)
plt.plot(fpr,fpr,"b--")
plt.grid(True,linewidth=0.5,linestyle="-.",alpha=0.5,color="grey")
plt.legend()
plt.show()


# In[92]:


f1=f1_score(y_test,lr2_pred)


# In[208]:


model.append("LR")
modelaccuracy.append(accuracy_score(y_test,lr2_pred))
modelf1.append(f1)
modelauc.append(roc_auc_lr)


# - - Decision Tree

# In[94]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(random_state=0)


# In[95]:


dt.fit(X_train,y_train)


# In[96]:


dt.score(X_train,y_train)       


# In[97]:


dt.score(X_test,y_test)


# **<u>Performance evaluation and optimizing parameters using GridSearchCV:**<u>

# In[98]:


parameters = {
    'max_depth':[1,2,3,4,5,None]
}


# In[99]:


gs_dt = GridSearchCV(dt, param_grid = parameters, cv=5, verbose=0)
gs_dt.fit(df_X_resample, df_y_resample)


# In[100]:


gs_dt.best_params_


# In[101]:


gs_dt.best_score_


# In[102]:


dt.feature_importances_


# In[103]:


dt2=DecisionTreeClassifier(max_depth=5)


# In[104]:


dt2.fit(X_train,y_train)
dt2_pred=dt2.predict(X_test)


# In[105]:


dt2.score(X_train,y_train)


# In[106]:


dt2.score(X_test,y_test)


# In[107]:


acuuracy_dt2=accuracy_score(y_test,dt2_pred)
f1_dt2=f1_score(y_test,dt2_pred)


# In[120]:


#ROC Curve
probs_dt2=dt2.predict_proba(X_test)
probs_dt2=probs_dt2[:,1]

fpr,tpr,threshold=roc_curve(y_test,probs_dt2)
roc_auc_dt2=auc(fpr,tpr)
plt.figure(figsize=(6,5),dpi=90)
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.title("ROC Curve")
plt.plot(fpr,tpr,"b",label="AUC Score=%0.3f"%roc_auc_dt2)
plt.plot(fpr,fpr,"r--")
plt.grid(True,color="grey",linestyle="-.",linewidth=0.5,alpha=0.5)
plt.legend()


# In[127]:


#Precision Recall Curve

pred_y_test = dt2.predict(X_test)                                     # predict class values
precision, recall, thresholds = precision_recall_curve(y_test, probs_dt2) # calculate precision-recall curve
f1 = f1_score(y_test, dt2_pred)                                    # calculate F1 score
auc_dt_pr = auc(recall, precision)                                    # calculate precision-recall AUC
ap = average_precision_score(y_test, probs_dt2)    # calculate average precision score
plt.figure(figsize=(6,5),dpi=80)
print('f1=%.3f auc_pr=%.3f ap=%.3f' % (f1, auc_dt_pr, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')                          # plot no skill
plt.plot(recall, precision, marker='.')                               # plot the precision-recall curve for the model
plt.xlabel("Recall")
plt.grid(True,linestyle="-.",linewidth=0.5,alpha=0.5)
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")


# In[209]:


model.append("Decision Tree")
modelaccuracy.append(accuracy_score(y_test,dt2_pred))
modelf1.append(f1)
modelauc.append(roc_auc_dt2)


# - - RandomForest Classifier

# In[137]:


from sklearn.ensemble import RandomForestClassifier
rf1=RandomForestClassifier(random_state=1)


# In[138]:


rf1.fit(X_train,y_train)
rf1_pred=rf1.predict(X_test)


# In[139]:


print(rf1.score(X_train,y_train))
rf1.score(X_test,y_test)


# **Performance evaluation and optimizing parameters using GridSearchCV:**

# In[140]:


parameters={"n_estimators":[50,100,150],
           "max_depth":[None,1,3,5],
           "min_samples_leaf":[1,3,5]
           }


# In[146]:


gs_rf1=GridSearchCV(estimator=rf1,param_grid=parameters,cv=5,verbose=0)
gs_rf1.fit(df_X_resample,df_y_resample)


# In[147]:


gs_rf1.best_params_


# In[148]:


gs_rf1.best_score_


# In[151]:


rf2=RandomForestClassifier(max_depth=None,min_samples_leaf=1,n_estimators=50)


# In[152]:


rf2.fit(X_train,y_train)
rf2_pred=rf2.predict(X_test)


# In[157]:


print("ROC Curve")
probs_rf2=rf2.predict_proba(X_test)
probs_rf2=probs_rf2[:,1]

fpr,tpr,threshold=roc_curve(y_test,probs_rf2)
roc_auc_rf2=auc(fpr,tpr)
print("AUC_Score=%3.2f"%roc_auc_rf2)
plt.figure(figsize=(6,5),dpi=90)
plt.plot(fpr,tpr,"b",label="AUC Score=%0.2f"%roc_auc_rf2)
plt.plot(fpr,fpr,"r--")
plt.grid(True,linestyle="-.",linewidth=0.5,alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()


# In[173]:


#Precision Recall Curve

precision,recall,threshold=precision_recall_curve(y_test,probs_rf2)
pr_auc_rf2=auc(recall,precision)

sns.set_style("whitegrid")
plt.figure(figsize=(6,5),dpi=90)
plt.plot(recall,precision,label="AUC Score=%0.2f"%pr_auc_rf2)
plt.plot([0,1],[0.5,0.5],"r--")
plt.legend(loc="best")
plt.xlabel("Recall")
plt.ylabel("Precision")


# In[210]:


model.append("RF")
modelaccuracy.append(accuracy_score(y_test,rf2_pred))
modelf1.append(f1_score(y_test,rf2_pred))
modelauc.append(roc_auc_rf2)


# - - - **<u>Support Vector Machine(SVM) Algorithm<u>**

# In[176]:


from sklearn.svm import SVC


# In[180]:


svm1=SVC(kernel="rbf")


# In[181]:


svm1.fit(X_train,y_train)
svm1_pred=svm1.predict(X_test)


# **<u>Performance evaluation and optimizing parameters using GridSearchCV:**<u>

# In[182]:


parameters = {
    'C':[1, 5, 10, 15, 20, 25],
    'gamma':[0.001, 0.005, 0.0001, 0.00001]
}


# In[184]:


gs_svm = GridSearchCV(estimator=svm1, param_grid=parameters, cv=5, verbose=0)
gs_svm.fit(df_X_resample, df_y_resample)


# In[185]:


gs_svm.best_params_


# In[186]:


gs_svm.best_score_


# In[187]:


svm2 = SVC(kernel='rbf', C=5, gamma=0.005, probability=True)


# In[188]:


svm2.fit(X_train, y_train)
svm2_pred=svm2.predict(X_test)


# In[190]:


#ROC Curve

svm2_probs=svm2.predict_proba(X_test)
svm2_probs=svm2_probs[:,1]

fpr,tpr,threshold=roc_curve(y_test,svm2_probs)
roc_auc_svm2=auc(fpr,tpr)
plt.figure(figsize=(6,5),dpi=90)
plt.plot(fpr,tpr,"b",label="AUC Score=%0.2f"%roc_auc_svm2)
plt.plot(fpr,fpr,"r--")
plt.xlabel("True Positive Rate")
plt.ylabel("False Positive Rate")
plt.legend()


# In[191]:


#Precision Recall Curve

precision,recall,threshold=precision_recall_curve(y_test,svm2_probs)
pr_auc_svm2=auc(recall,precision)

plt.figure(figsize=(6,5),dpi=90)
plt.plot(recall,precision,"b",label="AUC Score=%0.2f"%pr_auc_svm2)
plt.plot([0,1],[0.5,0.5],"r--")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision Recall Curve")
plt.legend()


# In[211]:


model.append("SVM")
modelaccuracy.append(accuracy_score(y_test,svm2_pred))
modelf1.append(f1_score(y_test,svm2_pred))
modelauc.append(roc_auc_svm2)


# In[214]:


model_summary = pd.DataFrame(zip(model,modelaccuracy,modelf1,modelauc), columns = ['model','accuracy','f1_score','auc'])
model_summary = model_summary.set_index('model')


# In[215]:


model_summary


# Overall, based on these metrics, the **<u>Random Forest<u>** model appears to be the best performing model among the given models for the given dataset.

# In[ ]:




