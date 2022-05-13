# Importing the libraries

from datetime import datetime
def current_time():
    return datetime.now()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
# exploring the dataset
dataset=pd.read_csv('SensorData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

import matplotlib.pyplot as plt
def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')

    fig.tight_layout()
    plt.show()
draw_histograms(dataset,dataset.columns,6,3)


dataset.Results.value_counts()
import seaborn as sn
sn.countplot(x='Results',data=dataset)

EEGRA_start_time = current_time()

#handling missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,4:6])
X[:,4:6]=imputer.transform(X[:,4:6])
#splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

# EEGRA using LR
from sklearn.linear_model import LogisticRegression
classifier =LogisticRegression()
classifier.fit(X_train,Y_train)

#Predict the test set results

y_Class_pred=classifier.predict(X_test)

#checking the accuracy for predicted results
from sklearn.metrics import accuracy_score
EEGRA_acc=accuracy_score(Y_test,y_Class_pred)

#Interpretation:

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_Class_pred))
#ROC foe EEGRA using LR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
EEGRA_roc = roc_auc_score(Y_test, classifier.predict(X_test))
EEGRA_fpr, EEGRA_tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(EEGRA_fpr, EEGRA_tpr, label='EEGRA (area = %0.2f)' % EEGRA_roc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('EEGRA_ROC')
plt.show()

EEGRA_end_time=current_time()

EEGRA_Process_time=EEGRA_end_time-EEGRA_start_time

##PREDICTION FOR NEW DATASET

Newdataset = pd.read_csv('newdata.csv')
ynew=classifier.predict(Newdataset)



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset=pd.read_csv('SensorData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 6].values

Proposed_start_time=current_time()
#handling missing data

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,4:6])
X[:,4:6]=imputer.transform(X[:,4:6])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#EXPLORING THE DATASET
import seaborn as sn
sn.countplot(x='Results',data=dataset)
dataset.Results.value_counts()
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#ACCURACY SCORE
from sklearn.metrics import accuracy_score
Proposed_m=accuracy_score(y_test,y_pred)

#Interpretation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


#ROC for proposed using NB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
proposed_roc_au = roc_auc_score(y_test, classifier.predict(X_test))
proposed_fpr, proposed_tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(proposed_fpr, proposed_tpr, label='Proposed (area = %0.2f)' % proposed_roc_au)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('EEGRA_ROC')
plt.show()

proposed_end_time=current_time()

proposed_ptime=proposed_end_time-Proposed_start_time
#PREDICTION FOR NEW DATASET

Newdataset = pd.read_csv('newdata.csv')
ynew=classifier.predict(Newdataset)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset=pd.read_csv('SensorData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:, 6].values

I_Leach_start_time=current_time()

#handling missing data

from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,4:6])
X[:,4:6]=imputer.transform(X[:,4:6])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#EXPLORING THE DATASET
import seaborn as sn
sn.countplot(x='Results',data=dataset)
dataset.Results.value_counts()
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
#ACCURACY SCORE
from sklearn.metrics import accuracy_score
I_Leach=accuracy_score(y_test,y_pred)

#Interpretation
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#ROC for I_LEACH using DT
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
dt_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
leach_fpr, leach_tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(leach_fpr, leach_tpr, label='I-Leach (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('EEGRA_ROC')
plt.show()

I_LEACH_endtime=current_time()

I_Leach_process_time=I_LEACH_endtime-I_Leach_start_time


##PREDICTION FOR NEW DATASET

dataset=pd.read_csv('SensorData.csv')
ynew=classifier.predict(Newdataset)


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset=pd.read_csv('SensorData.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 6].values

svc_start_time=current_time()

#handling missing data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(X[:,4:6])
X[:,4:6]=imputer.transform(X[:,4:6])

#splitting dataset into training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.25)

#feature scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


#EXPLORING THE DATASET
import seaborn as sn

sn.countplot(x='Results',data=dataset)
dataset.Results.value_counts()
##checking for different kernels

from sklearn.svm import SVC
svc_scores = []
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(kernels)):
    svc_classifier = SVC(kernel = kernels[i])
    svc_classifier.fit(X_train, Y_train)
    svc_scores.append(svc_classifier.score(X_test, Y_test))
from matplotlib.cm import rainbow
##%matplotlib inline
colors = rainbow(np.linspace(0, 1, len(kernels)))
plt.bar(kernels, svc_scores, color = colors)
for i in range(len(kernels)):
    plt.text(i, svc_scores[i], svc_scores[i])
plt.xlabel('Kernels')
plt.ylabel('Scores')
plt.title('Support Vector Classifier scores for different kernels')
#dataset.describe()
classifier = SVC(kernel = 'rbf', random_state = 0 ,probability=True)
classifier.fit(X_train, Y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
EBAR_acc=accuracy_score(Y_test,y_pred)

#Interpretation:

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))
#ROC for EBAR
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
EBAR_roc_auc = roc_auc_score(Y_test, classifier.predict(X_test))
EBAR_fpr, EBAR_tpr, thresholds = roc_curve(Y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(EBAR_fpr, EBAR_tpr, label='EBAR (area = %0.2f)' % EBAR_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('EEGRA_ROC')
plt.show()

svc_end_time=current_time()
EBAR_process_time=svc_end_time-svc_start_time
dataset=pd.read_csv('SensorData.csv')
ynew=classifier.predict(Newdataset)

print("EEGRA Accuracy :", EEGRA_acc) 
print("Proposed Accuracy :", Proposed_m) 
print("I-Leach Accuracy :", I_Leach)
print("EBAR Accuracy :", EBAR_acc)

#print("Processing times:\n EEGRA:{}\n Proposed:{}\n I-Leach: {}\n EBAR:{}".format(EEGRA_Process_time,proposed_ptime, I_Leach_process_time, EBAR_process_time))

x=['EEGRA','Proposed', 'I-Leach', 'EBAR' ]
y=[EEGRA_acc, Proposed_m, I_Leach, EBAR_acc]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='green')
plt.xlabel("Comparision of Algorithms")
plt.ylabel("Throughput")
plt.title("Throughput")

plt.xticks(x_pos, x)
plt.show()

fig = plt.figure(figsize=(14,11))

plt.subplot(2, 2, 1)
plt.plot(EEGRA_fpr, EEGRA_tpr, label='EEGRA (area = %0.2f)' % EEGRA_roc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.subplot(2, 2, 2)
plt.plot(proposed_fpr, proposed_tpr, label='Proposed (area = %0.2f)' % proposed_roc_au)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.subplot(2, 2, 3)
plt.plot(leach_fpr, leach_tpr, label='I-Leach (area = %0.2f)' % dt_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")

plt.subplot(2, 2, 4)
plt.plot(EBAR_fpr, EBAR_tpr, label='EBAR (area = %0.2f)' % EBAR_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



x=['EEGRA','Proposed', 'I-Leach', 'EBAR' ]
y=[EEGRA_acc, Proposed_m, I_Leach, EBAR_acc]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='blue')
plt.xlabel("Comparision of Algorithms")
plt.ylabel("Average Residual Energy")
plt.title("Average residual energies")

plt.xticks(x_pos, x)
plt.show()

#Processing Times Plot
x=['EEGRA','Proposed', 'I-Leach', 'EBAR' ]
y=[int(str(EEGRA_Process_time)[-6:]), int(str(proposed_ptime)[-6:]), int(str(I_Leach_process_time)[-6:]), int(str(EBAR_process_time)[-6:]) ]
x_pos = [i for i, _ in enumerate(x)]

plt.bar(x_pos, y, color='green')
plt.xlabel("Classifier ")
plt.ylabel("Time taken in microseconds")
plt.title("Processing Times")

plt.xticks(x_pos, x)

plt.show()