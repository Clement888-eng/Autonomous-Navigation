from math import log2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import PolynomialFeatures as poly
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def gini(data):
    ans=1
    for i in data:
        ans-=i**2
    print(f'Q is {ans}')

def entropy(data):
    ans=0
    for i in data:
        if i==0:
            continue
        else:
            ans-=(i*log2(i))
    print(f'Q is {ans}')

def misclassification(data):
    ans=1-max(data)
    print(f'Q is {ans}')

def var(data):
    y_bar=np.mean(data)
    total=0
    for i in data:
        total+=(i-y_bar)**2
    ans=total/len(data)
    print(f'MSE is {ans}')
    
def decision_tree_accuracy(X,y,random,depth,test,crit):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test,random_state=random)
    regressor=tree.DecisionTreeClassifier(criterion=crit, max_depth=depth, random_state=random)
    regressor.fit(X_train,y_train)
    ytr_pred=regressor.predict(X_train)
    yts_pred=regressor.predict(X_test)
    acc_ytr=acc(y_train,ytr_pred)
    acc_yts=acc(y_test,yts_pred)
    print(f'Accuracy test = {acc_yts}')
    print(f'Accuracy train = {acc_ytr}')

def evaluation_metrics(TP,FP,TN,FN,out):
    if out=='TPR':
        return TP/(TP+FN)
    elif out=='FNR':
        return FN/(TP+FN)
    elif out=='TNR':
        return TN/(FP+TN)
    elif out=='FPR':
        return FP/(FP+TN)

def gradient_descent(initial,learning_rate,iteration):
    #modify function here
    x=initial
    for i in range(iteration):
        y=x**4
        dy=4*(x**3)
        x=x-learning_rate*dy
    return x

def regr_without_reg(X_train,y_train,X_test,deg):
    p=poly(degree=deg)
    X_train_poly=p.fit_transform(X_train)
    model=LinearRegression()
    model.fit(X_train_poly,y_train)
    X_test_poly=p.fit_transform(X_test)
    y_pred=model.predict(X_test_poly)
    return y_pred

def regr_with_reg(X_train,y_train,to_predict,deg,a):
    rr=Ridge(alpha=a)
    p=poly(degree=deg)
    X_train_poly=p.fit_transform(X_train)
    to_predict_poly=p.fit_transform(to_predict)
    rr.fit(X_train_poly,y_train)
    y_pred=rr.predict(to_predict_poly)
    return y_pred

def average_KFold(X,y,fold,start_deg,finish_deg,error_type):
    kf=KFold(n_splits=fold)
    error_test,error_train=[],[]
    KFold(n_splits=5,random_state=None, shuffle=False)
    for i in range(start_deg,finish_deg+1):
        err1,err2=0,0
        for train_index, test_index in kf.split(X):
            X_train,X_test=X[train_index],X[test_index]
            y_train,y_test=y[train_index],y[test_index]
            P=poly(i)
            P_train=P.fit_transform(X_train)
            a=np.linalg.inv(P_train.transpose().dot(P_train))
            w=a.dot(P_train.transpose())
            W=w.dot(y_train)
            P_test=P.fit_transform(X_test)
            P_train=P.fit_transform(X_train)
            prediction_test=P_test.dot(W)
            prediction_train=P_train.dot(W)
            err1+=MSE(y_test,prediction_test)
            err2+=MSE(y_train,prediction_train)
        err1=err1/fold
        err2=err2/fold
        error_test.append(err1)
        error_train.append(err2)
    return error_test,error_train

def decision_tree(X_train,y_train,X_test,crit,depth,random):
    regr=DecisionTreeClassifier(criterion=crit,max_depth=depth, random_state=random)
    regr.fit(X_train,y_train)
    y_pred=regr.predict(X_test)
    return y_pred

def find_accuracy_score(y_true,y_pred):
    accuracy_score(y_trus,y_pred)

def kmeans(data,clusters,to_predict):
    color=['red','green','yellow','blue','cyan','magenta','grey']
    kmeans=KMeans(n_clusters=clusters, random_state=0).fit(data)
    y_pred=kmeans.fit_predict(data)
    for i in range(clusters):
        plt.scatter(data[y_pred==i,0],data[y_pred==i,1],s=5,c=color[i])
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=30,c='black')
    plt.show()














    
