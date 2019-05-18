# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 06:23:11 2019

@author: Gupeng
"""



'''
待解决的问题：
1. 是否使用K折交叉验证

2. 是否需要正则化  部分数据需要正则化 

'''
#import sklearn as sk
import sys,os
os.getcwd()
#sys.path.append("F:\\DataDig\\Intern\options")
#from fileopt import readfile as read
import pandas as pd

from sklearn.preprocessing import StandardScaler
# utils是utilities的缩写。意思是小工具
# 该模块实现数据嵌入技术
#from sklearn.manifold import TSNE
from uitls import plot_confuseMatrix,smote,pcaview

    #将Amount 标准化，去除 Class和Time类别 因为这两个类别数值太大 意义不大
def splitdata(path="F:\\mldata\creditcard.csv"):
    df = pd.read_csv(path)
#    标准化Amount 否则数值太大，影响模型 
    df['normAmount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
    df=df.drop('Amount',axis=1)
    fraud = df[df.Class == 1]
    normal = df[df.Class == 0]
    x_train=fraud.sample(frac=0.7)
    x_train = pd.concat([x_train, normal.sample(frac=0.7)])
    x_test = df.loc[~df.index.isin(x_train.index)]    
    y_train = x_train.Class
    y_test=x_test.Class
    x_train = x_train.drop(['Class', 'Time'], axis=1)
    x_test = x_test.drop(['Class', 'Time'], axis=1)
    return df,x_train,pd.DataFrame(y_train),x_test,pd.DataFrame(y_test)

from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import KFold, cross_val_score


#折交叉验证


#正则化

def narmAmount(data):
    
    data=data.drop()
    return 
    
#逻辑回归    
def lrtrain(x_train,y_train):
    lrmodel = LogisticRegression(penalty='l2',solver='lbfgs')
    lrmodel.fit(x_train, y_train.values.ravel())      
    return lrmodel
def lrtest(lrmodel,x_test,y_test):
    y_pre= lrmodel.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score
    
#决策树
from sklearn.tree import DecisionTreeClassifier
def dttrain(x_train, y_train):
    model=DecisionTreeClassifier(criterion='entropy')
    model.fit(x_train, y_train.values.ravel())
    return model
def dttest(model,x_test,y_test):
    y_pre=model.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score

#随机森林 sk实现
from sklearn.ensemble import RandomForestClassifier
def rfauto(x_train, y_train):    
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train.values.ravel())
    return rfc
def rfautotest(model,x_test,y_test):
    y_pre = model.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score
#随机森林（手工实现） 
def rftrain(ntree,x_train,y_train):
    modellist=[]
    for i in range(ntree):
        x_trainchild=x_train.sample(frac=0.7)
        y_trainchild=y_train[y_train.index.isin(x_trainchild.index)]
        x_trainchild=x_trainchild.sample(frac=0.7,axis=1)
        model=dttrain(x_trainchild,y_trainchild)
        modellist.append(model)
    return modellist     

#支持向量机    
from sklearn import svm
#还有 参数的确定？？
def svmtrain(x_train,y_train):
#    prob = svm_problem(y_train, x_train)
#    param = svm_parameter('-t 0 -c 4 -b 1')
#    model = svm_train(prob, param)
#    p_label, p_acc, p_val = svm_predict(y_test, x_test, model)
#    return p_label, p_acc, p_val
    
    cls = svm.LinearSVC()
    cls.fit(x_train,y_train.values.ravel())
    return cls
def svmtest(model,x_test,y_test):
    y_pre=model.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score

#GBDT
from sklearn.ensemble import GradientBoostingClassifier
def gbdttrain(x_train,y_train):
    gbdt=GradientBoostingClassifier(
  loss='deviance'
, learning_rate=0.1
, n_estimators=100
, subsample=1
, min_samples_split=2
, min_samples_leaf=1
, max_depth=4
)
    gbdt.fit(x_train,y_train.values.ravel())
    return gbdt
def gbdttest(model,x_test,y_test):
    y_pre=model.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score
#XGBOOST
from xgboost.sklearn import XGBClassifier
#    paras={
#        'booster':'gbtree',
##        'objective':'multi:softmax',
#        'objective':'binary:logistic',
##        'num_class':2,
#        'gamma':0.05,#树的叶子节点下一个区分的最小损失，越大算法模型越保守
#        'max_depth':12,
#        'lambda':450,#L2正则项权重
#        'subsample':0.4,#采样训练数据，设置为0.5
#        'colsample_bytree':0.7,#构建树时的采样比率
#        'min_child_weight':12,#节点的最少特征数
#        'silent':1,
#        'eta':0.005,#类似学习率
#        'seed':700,
#        'nthread':4,#cpu线程数
#    }
#    plst=list(paras.items())#超参数放到集合plst中;
##    offset=35000#训练集中数据50000,划分35000用作训练，15000用作验证
#    num_rounds=500#迭代次数
#    xgtrain=xgb.DMatrix(x_train,y_train)#将训练集的二维数组加入到里面
#    print (xgtrain)
##    watchlist =[(xgtrain,'train'),(xgval,'val')]#return训练和验证的错误率
##    xgb.train(plst,xgtrain,num_rounds,watchlist,early_stopping_rounds=100)
#    #eval 参数用于观测
#    model = xgb.train(plst,xgtrain,num_rounds,early_stopping_rounds=100)
#    return model
def xgbtrain(x_train,y_train):    
    clf = XGBClassifier(
    silent=1 ,#设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    #nthread=4,# cpu 线程数 默认最大
    learning_rate= 0.3, # 如同学习率
    min_child_weight=1, 
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    max_depth=6, # 构建树的深度，越大越容易过拟合
    gamma=0.1,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    subsample=1, # 随机采样训练样本 训练实例的子采样比
    max_delta_step=0,#最大增量步长，我们允许每个树的权重估计。
    colsample_bytree=1, # 生成树时进行的列采样 
    reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    #reg_alpha=0, # L1 正则项参数
    scale_pos_weight=1, #如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重
    #objective= 'multi:softmax', #多分类的问题 指定学习任务和相应的学习目标
    #num_class=10, # 类别数，多分类与 multisoftmax 并用
    n_estimators=100, #树的个数
#    seed=0 #随机种子
    #eval_metric= 'auc'
 )
    clf.fit(x_train,y_train.values.ravel(),eval_metric='auc')
    return clf

def xgbtest(model,x_test,y_test):
#    xgtest=xgb.DMatrix(x_test)
    #ntreelimits : 如果提前结束了训练，那么在预测时就使用最佳迭代次数
    #If early stopping is enabled during training, you can predict with the best iteration.
    #y_pre=model.predict(x_test,ntree_limit=model.best_iteration)
    y_pre=model.predict(x_test)
    score=plot_confuseMatrix(y_test,y_pre)
    return y_pre,score
#LIGHTGBM
import lightgbm

#过采样

#GAN网络与强化学习 


if __name__ == '__main__':
    df,x_train,y_train,x_test,y_test=splitdata() 
    x_pca=pcaview(x_train,y_train)
    x_trainsam,y_trainsam=smote(x_train,y_train,1000)
    x_pca=pcaview(x_trainsam,y_trainsam)
#    auclist=[]
#    print("LR")
#    model=lrtrain(x_train,y_train)
#    y_pre,score1=lrtest(model,x_test,y_test)   
#    auclist.append(score1)
#    print("SVM")
#    model=svmtrain(x_train,y_train)
#    y_presvm,score1=svmtest(model,x_test,y_test)
#    auclist.append(score1)
#    print("SMOTELR")
#    modelsam=lrtrain(x_trainsam,y_trainsam)
#    y_presam,score1=lrtest(modelsam,x_test,y_test)
#    auclist.append(score1)
#    print("SMOTESVM")
#    modelsam=svmtrain(x_trainsam,y_trainsam)
#    y_presam,score1=svmtest(modelsam,x_test,y_test)
#    auclist.append(score1)
    
    
    
    
    
    
    

