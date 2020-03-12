import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import os
import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
f = 'E:\wage'
os.chdir(f)
##train_path = "E:\\wage\\adult.data"##此处为双斜杠
#test_path = "E:\\wage\\adult.test.csv"
train_set = pd.read_csv('adult_data.csv', header=None)
#print(train_set.head())
test_set=pd.read_csv('adult_test.csv', header=None)
#print(train_set.info()) #32561
#print(test_set.info()) #16281
col_labels = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
              'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
              'hours_per_week', 'native_country', 'wage_class']

train_set.columns = col_labels
test_set.columns = col_labels
test_set['wage_class'] = test_set.wage_class.replace({' <=50K.': ' <=50K', ' >50K.': ' >50K'})
train_set=train_set.replace(' ?', np.nan)
test_set=test_set.replace(' ?', np.nan)
'''
total=test_set.isnull().sum().sort_values(ascending=False)
percent=(test_set.isnull().sum()/test_set.isnull().count()).sort_values(ascending=False)
print(total,percent)
'''

#c1=train_set['native_country'].isnull().index
#train_set=train_set.drop(c1)
train_set["native_country"]=train_set["native_country"].fillna("None")
train_set["workclass"]=train_set["workclass"].fillna("None")
train_set["occupation"]=train_set["occupation"].fillna("None")
test_set["native_country"]=test_set["native_country"].fillna("None")
test_set["workclass"]=test_set["workclass"].fillna("None")
test_set["occupation"]=test_set["occupation"].fillna("None")

combined_set = pd.concat([train_set, test_set], axis=0)
for feature in combined_set.columns:
    if combined_set[feature].dtype == 'object':
        combined_set[feature] = pd.Categorical(combined_set[feature]).codes

train_set = combined_set[:train_set.shape[0]]
test_set = combined_set[train_set.shape[0]:]
print(test_set.head())
X_train=train_set.loc[:,'age':'native_country']
y_train=train_set.loc[:,'wage_class']
X_test=test_set.loc[:,'age':'native_country']
y_test=test_set.loc[:,'wage_class']

X=X_train
Y=y_train
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.3, random_state=0)

num=1
kf = KFold(n_splits=10)
for train, test in kf.split(X):
    X_train, X_valid = X.loc[train], X.loc[test]
    Y_train, Y_valid = Y.loc[train], Y.loc[test]
    rf = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=400)
    rf.fit(X_train, Y_train)
    valid_predictions = rf.predict(X_valid)
    accuracy = accuracy_score(Y_valid, valid_predictions)
    joblib.dump(rf, "rf.m")
    print("随机森林"+str(num)+"验证集准确率:  %s " % accuracy)
    num = num + 1

'''
# 构建随机森林模型
rf = RandomForestClassifier(oob_score=False, random_state=10, criterion='entropy', n_estimators=1000)
rf.fit(X_train, Y_train)
joblib.dump(rf, "rf.m")
# 调参过程可采用程序控制,使用验证集准确率调参
validation_predictions = rf.predict(X_validation)
print("验证集准确率:  %s " % accuracy_score(Y_validation, validation_predictions))
'''

rf = joblib.load("rf.m")
Y_predictions1 = rf.predict(X_test)
confmat = confusion_matrix(y_test, Y_predictions1)
print(confmat)
print(classification_report(y_test, Y_predictions1,digits=3))






