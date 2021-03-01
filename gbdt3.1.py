import random
from pandas import read_csv, np
from sklearn import linear_model, metrics
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals.joblib import dump
from sklearn.model_selection import  train_test_split
from sklearn.externals.joblib import  load
from matplotlib import pyplot as plt

import time

def plot_confusion_matrix(cm,labels_name,title):
    cm=cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm,interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_local=np.array(range(len(labels_name)))
    plt.xticks(num_local,labels_name,rotation=90)
    plt.yticks(num_local,labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def wgn(X,snr):
    P_signal=np.sum(abs(X)**2)/(len(X))
    P_noise=P_signal/10**(snr/10.0)
    #return np.random.randn(X.shape[1])*np.sqrt(P_noise)
    return np.random.randn(len(X)) * np.sqrt(P_noise)


start = time.time()
filename='C:/Users/lenovo/Desktop/signal-160.csv'
data160=read_csv(filename)

array160=data160.values
print(array160.shape)

X160=array160[:,4000:13000]
print(X160.shape)
Y160=array160[:,0:1].ravel()

filename='C:/Users/lenovo/Desktop/signal-320.csv'
data320=read_csv(filename)
array320=data320.values
X320=array320[:,4000:13000]
Y320=array320[:,0:1].ravel()

filename='C:/Users/lenovo/Desktop/signal-480.csv'
data480=read_csv(filename)
array480=data480.values
X480=array480[:,4000:13000]
Y480=array480[:,0:1].ravel()

filename='C:/Users/lenovo/Desktop/signal-640.csv'
data640=read_csv(filename)
array640=data640.values
X640=array640[:,4000:13000]
Y640=array640[:,0:1].ravel()

filename='C:/Users/lenovo/Desktop/signal-800.csv'
data800=read_csv(filename)
array800=data800.values
X800=array800[:,4000:13000]
Y800=array800[:,0:1].ravel()

X=np.vstack((X160,X320,X480,X640,X800))
Y=np.hstack((Y160,Y320))
Y=np.hstack((Y,Y480))
Y=np.hstack((Y,Y640))
Y=np.hstack((Y,Y800))
print(X.shape)
print(Y.shape)
'''
plt.title('signal60')
plt.plot(X[59])
plt.show()
'''
#noise=wgn(X,0)
#X_noise=X+noise
for i in range (796):
#X[i]+=random.gauss(0,0.18)
    X[i]+=wgn(X[i],40)


plt.title('noised signal60 (snr=9db)')
plt.plot(X[59])
plt.show()

#Y=array[:,0:1].ravel()
#print(Y.shape)


#split
test_size=0.33
seed=6
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size,random_state=seed)

#train&test
model=GradientBoostingClassifier(n_estimators=100)#maximum number of weak learners
model.fit(X_train,Y_train)

#model score
pred_Y = model.predict(X_test)
result = model.score(X_test,Y_test)
print("algorithm evaluation result:%.3f%%"%(result*100))

#confusion_matrix
print("confusion_matrix:")
cm=metrics.confusion_matrix(Y_test,pred_Y)
print(cm)
labels_name=['1st','2nd','3rd','4th','5th','6th','7th','8th']
plot_confusion_matrix(cm,labels_name,"GBDT Confusion Matrix (snr=9db)")
plt.show()


#print("precision recall curve:"+precision_recall_curve( ,pred_Y))
#print("roc_curve:")
#fpr,tpr,thresholds=metrics.roc_curve(Y_test,pred_Y,pos_label=2)
#print(fpr,'\n',tpr,'\n',thresholds)
'''
print("roc_curve:")
fpr=dict()
tpr=dict()
roc_auc=dict()
n_classes=Y.shape[1]
for i in range(n_classes):
    fpr[i],tpr[i],_=roc_curve(Y_test[:,i],pred_Y[:,i])
    roc_auc[i]=auc(fpr[i],tpr[i])

fpr["micro"],tpr["micro"],_=roc_curve(Y_test.ravel(),pred_Y.ravel())
roc_auc["micro"]=auc(fpr["micro"],tpr["micro"])
plt.figure()
lw=2
plt.plot(fpr[2],tpr[2],color='darkoraange',lw=lw,label='ROC curve(area=%0.2f")'%roc_auc[2])
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.title('ROC_curve1')
plt.xlabel('false presitive rate')
plt.ylabel('true presitive rate')
plt.ylim(0,1.05)
plt.xlim(0,1.05)
plt.legend(loc="lower right")
plt.show()
'''
end=time.time()
print(str(end))
