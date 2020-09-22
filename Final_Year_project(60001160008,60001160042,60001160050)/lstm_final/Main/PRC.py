#!/usr/bin/env python
# coding: utf-8

# In[1]:

def feat_imp():
    

    import pandas as pd
    import numpy as np
    import talib as ta
    import matplotlib.pyplot as plt
    #get_ipython().run_line_magic('matplotlib', 'inline')
    #from nsepy import get_history
    from datetime import date
    import LSTM9
    from LSTM9 import CSV_file


    # In[2]:


    df=pd.read_csv(LSTM9.CSV_file+'.csv',index_col='Date',parse_dates=True)

    #df=pd.read_csv('995_PAGE_IND.csv',index_col='datetime',parse_dates=True)
    #df=pd.read_csv('995_PAGE_IND.csv',index_col='datetime',parse_dates=True)
    #del df['Volume']
    #del df['Turnover (Lacs)']
    O=df['Open']
    H=df['High']
    L=df['Low']
    C=df['Close']


    # In[3]:


    df['ADX'] = ta.ADX(O, L, C, timeperiod=14)
    df['RSI'] = ta.RSI( C, timeperiod=14)


    df['SMA_10']=C.rolling(10).mean()
    df['SMA_21']=C.rolling(21).mean()
    df['SMA_3']=C.rolling(3).mean()
    df['SMA_50']=C.rolling(50).mean()


    df['exp_10'] = C.ewm(span=10, adjust=False).mean()
    df['exp_21'] = C.ewm(span=21, adjust=False).mean()


    #Average True Range
    df['ATR']=ta.ATR(H,L,C,timeperiod=14)




    #Commodiy Channel Index 
    df['CCI']=ta.CCI(H,L,C,timeperiod=14)


    #Momentum
    df['MOM']=ta.MOM(C,timeperiod=10)


    #ROC
    df['ROC']=ta.ROC(C,timeperiod=10)


    #ROCP
    df['ROCR']=ta.ROCP(C,timeperiod=10)



    #Williams %R

    df['Williams %R']=ta.WILLR(H,L,C,timeperiod=14)


    #Stochastic %K


    df['slowk'], df['slowd'] = ta.STOCH(H, L, C, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)


    # In[4]:


    #drop all the nan values
    df=df.dropna()
    df.head()


    #For Converting into a Binary Classification Problem we will assign them into 1 or 0 values
    df['pred_price']=np.where(df['Close'].shift(-1)>df['Close'],1,0)


    A=df['pred_price'].unique()

    print(A)


    # In[5]:


    #Seperating into target and test set

    y=df['pred_price']
    x=df.drop(columns=['pred_price'])



    #define train/test split ratio
    split_ratio=0.9


    train_x=x[0:int(split_ratio*len(x))]
    test_x=x[int(split_ratio*len(x)):(len(x))]



    print('Observations: %d' % (len(x)))
    print('Train Dataset:',train_x.shape)
    print('Test Dataset:',test_x.shape)


    print('----------------------------')


    train_y=y[0:int(split_ratio*len(y))]
    test_y=y[int(split_ratio*len(y)):(len(y))]



    print('Observations: %d' % (len(y)))
    print('Train Dataset:',train_y.shape)
    print('Test Dataset:',test_y.shape)










    # In[6]:


    #Normalizing the data


    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1)) #scaling Down
    train_x_scaled=scaler.fit_transform(train_x)

    test_x_scaled=scaler.fit_transform(test_x)

    print(train_x_scaled)


    # In[22]:


    import time 

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_moons, make_circles, make_classification
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression



    dict_classifiers={
        "Logistic Regression":LogisticRegression(solver='lbfgs',max_iter=5000),
        "Nearest Neighbors" : KNeighborsClassifier(),
        "Support Vector Machine": SVC(gamma='auto'),
        #"Gradient Boosting Classifier": XGBClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest":RandomForestClassifier(n_estimators=100),
        "Neural Net":MLPClassifier(solver='adam',alpha=0.0001,learning_rate='constant',learning_rate_init=0.001),
        "Naive Bayes":GaussianNB()
    }


    # In[23]:


    no_classifiers=len(dict_classifiers.keys())
    import time 


    def batch_classify(train_x_scaled,train_y,verbose=True):
        df_results=pd.DataFrame(data=np.zeros(shape=(no_classifiers,3)),columns=['classifier','train_score','training_time'])

        
        
        count=0
        for key,classifier in dict_classifiers.items():
            t_start=time.process_time()
            classifier.fit(train_x_scaled,train_y)
            t_end=time.process_time()
            t_diff=t_end-t_start

            train_score=classifier.score(train_x_scaled,train_y)
            df_results.loc[count,'classifier']=key
            df_results.loc[count,'train_score']=train_score
            df_results.loc[count,'training_time']=t_diff
            if verbose:
                print('trained {c} in {f:2f}s'.format(c=key,f=t_diff))
            count+=1
        return df_results
        


    # In[24]:


    df_results=batch_classify(train_x_scaled,train_y)
    print(df_results)


    # In[25]:


    log_reg = LogisticRegression(solver='lbfgs', max_iter=5000)
    log_reg.fit(train_x_scaled, train_y)


    # In[26]:


    import sklearn
    predictions=log_reg.predict(test_x_scaled)
    print('accuracy:',sklearn.metrics.accuracy_score(test_y,predictions))
    print("confusion matrix:",sklearn.metrics.confusion_matrix(test_y,predictions))
    print("classification report:",sklearn.metrics.classification_report(test_y,predictions))


    # In[27]:


    #roc curve


    y_pred_proba=log_reg.predict_proba(test_x_scaled)[:,1]

    fpr,tpr,thresholds=sklearn.metrics.roc_curve(test_y,y_pred_proba)

    roc_auc=sklearn.metrics.auc(fpr,tpr)


    #plot of Roc
    print('ROC AUC is ' +str(roc_auc))
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr,tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show(block=False)
    plt.pause(5)
    plt.close()


    # In[28]:


    #Feature Importance

    feature_importance=abs(log_reg.coef_[0])
    print(feature_importance)
    feature_importance=100.0*((feature_importance)/feature_importance.max())
    sorted_idx=np.argsort(feature_importance)
    pos=np.arange(sorted_idx.shape[0])+0.5

    featfig=plt.figure(figsize=(20,10))
    featax=featfig.add_subplot(1,1,1)
    featax.barh(pos,feature_importance[sorted_idx],align='center')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(train_x.columns)[sorted_idx],fontsize=15)
    featax.set_xlabel('Relative Feature Importance')

    plt.tight_layout()
    plt.show()


    # In[ ]:





    # In[ ]:





    # In[ ]:





