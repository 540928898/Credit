# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:29:06 2019

@author: Gupeng
"""

import sys
import basefunc as bf
import numpy as np
import pandas as pd
sys.path.append(
        'F:\\DataDig\\CreditFind\\improved_wgan_trainin')

import gan_credit as gc
from uitls import pcaview
from basefunc import lrtrain,lrtest
#lib.print_model_settings(locals().copy())
def data_gen(x_train,y_train):
    #合并
    xx=pd.concat([x_train,y_train],axis=1)
    datatrue=xx[xx.Class == 1]
    datafal=xx[xx.Class == 0]
    datatrue = datatrue.drop(['Class'], axis=1)
    datafal = datafal.drop(['Class'], axis=1)
    while True:
            dataset = []
#                for enum,evalue in enumerate(centers):
#                    point[enum]+=evalue                                                   
            dataset.extend(np.array(datatrue.sample(1)))
            dataset.extend(np.array(datafal.sample(1)))
            dataset = np.array(dataset, dtype='float32')
            yield dataset
if __name__ == '__main__':
    df,x_train,y_train,x_test,y_test=bf.splitdata()
#    ss=data_gen(x_train,y_train)
#    print(ss.__next__().shape)
    fakedata,fakelabel=gc.fit(data_gen,x_train,y_train)
    pd_fakedata=pd.DataFrame(fakedata,columns=x_train.columns)
    pd_fakelabel=pd.DataFrame(fakelabel,columns=y_train.columns)
    xx2=pd.concat([x_train,pd_fakedata])
    xx3=pd.concat([y_train,pd_fakelabel])
    before=pcaview(x_train,y_train)
    afterall=pcaview(xx2,xx3)
    model=lrtrain(x_train,y_train)
    y_pre,score1=lrtest(model,x_test,y_test)   
    model=lrtrain(xx2,xx3)
    y_pre2,score2=lrtest(model,x_test,y_test)  
