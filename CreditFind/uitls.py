# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 00:27:09 2019

@author: Gupeng
"""

#采样方法：
import pandas as pd
import matplotlib.gridspec as gridspec
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve,
                             confusion_matrix,
                             auc, roc_auc_score,
                             roc_curve, recall_score,
                             classification_report)
#可视化函数

#散点图

#混淆矩阵

#ROC 和AUC 

#TP（True Positive）：指正确分类的正样本数，即预测为正样本，实际也是正样本。
#FP（False Positive）：指被错误的标记为正样本的负样本数，即实际为负样本而被预测为正样本，所以是False。
#TN（True Negative）：指正确分类的负样本数，即预测为负样本，实际也是负样本。
#FN（False Negative）：指被错误的标记为负样本的正样本数，即实际为正样本而被预测为负样本，所以是False。
#TP+FP+TN+FN：样本总数。
#TP+FN：实际正样本数。
#TP+FP：预测结果为正样本的总数，包括预测正确的和错误的。
#FP+TN：实际负样本数。
#TN+FN：预测结果为负样本的总数，包括预测正确的和错误的。
#
#
#FPR=FP/(FP+TN)(实际的负样本中有多少被预测为了正样本，虚警率)
#TPR=TP/(TP+FN)(判断正确的正样本的比率，命中率)


def rocandauc(y_test,y_pre):
    actual = y_test
    predictions = y_pre
    false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, predictions)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    result=roc_auc_score(y_test, y_pre)
    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b',
    label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.2])
    plt.ylim([-0.1,1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return result

#预画图

#df.info()
def plotdata(df):
    #时间与行为的关系
    plt.figure(11)
    plt.subplot(2, 1, 1)
    plt.suptitle('Time')
    plt.hist(df.Time[df.Class == 1], bins=50)
    plt.title('fraud')
    plt.ylabel('transaction numbers')
    plt.subplot(212)
    plt.hist(df.Time[df.Class == 0], bins=50)
    plt.title('normal')
    plt.subplots_adjust(wspace =0, hspace =0.5) #调整间距
    
    #数量与实践的关系图
    plt.figure(12)
    plt.subplot(2, 1, 1)
    plt.suptitle('Amount')
    plt.subplot(211)
    plt.hist(df.Amount[df.Class == 1], bins=30)
    plt.title('fraud')
    plt.subplot(212)
    plt.hist(df.Amount[df.Class == 0], bins=30)
    plt.title('normal')
    plt.subplots_adjust(wspace =0, hspace =0.5)
    
    #其他因素 欺诈与非欺诈的关系 （核函数）
    features = [x for x in df.columns
                if x not in ['Time', 'Amount', 'Class']]
    plt.figure(figsize=(12, 28*4))
    # 隐式指定网格行数列数（隐式指定子图行列数）
    gs = gridspec.GridSpec(28, 1)
    for i,num in enumerate(features):
        ax = plt.subplot(gs[i])
        sns.distplot(df[num][df.Class == 1], bins=50, color='red')
        sns.distplot(df[num][df.Class == 0], bins=50, color='green')
        ax.set_title(str(num))
    plt.subplots_adjust(wspace =0, hspace =0.5)#调整子图间距              
    plt.savefig('各个变量与class的关系.png', transparent=False, bbox_inches='tight')
    

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler,scale,normalize


#降维视图
import numpy as np
def pcaview(data,label):
    model_pca = PCA(n_components=2)
    X_pca = model_pca.fit(data).transform(data)
    print(data.shape)
    print(data[0:5])
    print(X_pca.shape)
    print(X_pca[0:5])
    print("降维后各主成分方向：\n",model_pca.components_)
    print("降维后各主成分的方差值：",model_pca.explained_variance_)
    print("降维后各主成分的方差值与总方差之比：",model_pca.explained_variance_ratio_)
    print("奇异值分解后得到的特征值：",model_pca.singular_values_)
    print("降维后主成分数：",model_pca.n_components_)
    
    X_pca=pd.DataFrame(X_pca,columns=["X1","X2"])
    X_pca['X2norm'] = StandardScaler().fit_transform(X_pca['X2'].values.reshape(-1,1))

    X_pca['X1norm'] = StandardScaler().fit_transform(X_pca['X1'].values.reshape(-1,1))
  #    X_pca=X_pca.drop(['X1', 'X2'], axis=1)
    labelreset=label.reset_index(drop=True)
    xx=pd.concat([X_pca,labelreset],axis=1)
    #找出label为1 的数据
    datatrue=xx.loc[xx.Class == 1]
    datafalse=xx.loc[xx.Class == 0]
    plt.scatter(x="X1norm",y="X2norm",c="r",marker=".",data=datafalse,alpha=0.5)
    plt.scatter(x="X1norm",y="X2norm",marker=".",data=datatrue,alpha=0.5)
    plt.show()
#    sns.jointplot(x="X1",y="X2",data=datatrue)
#    sns.jointplot(x="X1",y="X2",data=datafalse)
    return X_pca



def plot_confuseMatrix(y_test,y_pre):
    print("混淆矩阵为： ")
    confusion_matrix(y_test, y_pre)
    print(classification_report(y_test, y_pre))
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pre),annot=True,cmap=plt.cm.Blues)
    aucscore=rocandauc(y_test,y_pre)    
    return aucscore
    
    

#下采样 
def lower_sample(x_data,y_data):
    data=pd.concat([x_data,y_data],axis=1)
    datatrue=data[data.Class==1]
    datafal=data[data.Class==0]
    datanew=pd.concat([datatrue,datafal.sample(frac=0.7)])
    xsample=datanew.loc[:,datanew.columns[:-1]]
    ysample=datanew.Class
    return xsample,ysample

#重采样
from imblearn.over_sampling import RandomOverSampler

from collections import Counter
#过采样 容易过拟合 不使用 
def over_sample(x_data,y_data):
#    data=pd.concat([x_data,y_data],axis=1)
    ros = RandomOverSampler(random_state=1)
    x_resampled, y_resampled = ros.fit_sample(x_data, y_data)
    print(Counter(y_resampled))
    return x_resampled, y_resampled

#ENN
def ennsample():
    pass

#RENN
def rennsample():
    pass

#smote    
from imblearn.over_sampling import SMOTE
def smote(x_data,y_data,num):
    print("未生成前数据：", Counter(y_data.Class))
    smo = SMOTE(ratio={1:num},random_state=42)
    x_smo, y_smo = smo.fit_sample(x_data, y_data.values.ravel())
    print("生成数据后的：", Counter(y_smo))
    return pd.DataFrame(x_smo,columns=x_data.columns),pd.DataFrame(y_smo.ravel(),columns=y_data.columns)

#ADASYN


