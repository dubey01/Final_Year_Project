#!/usr/bin/env python
# coding: utf-8

# Import all the necesssary libraries
# 
# 

# In[18]:
CSV_file = ""
def neural_N():
    import math
    #import pandas_datareader as web
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense,LSTM,Dropout
    import matplotlib.pyplot as plt
    import seaborn as sns

    from matplotlib import style
    #from mpl_finance import candlestick_ohlc
    import matplotlib.dates as mdates
    #import pandas_datareader as web





    
    #import keras
    #import sklearn
    sns.set()


    # In[19]:


    df = pd.read_csv(CSV_file+'.csv')
    df.head()


    # In[20]:


    plt.plot((df['Close']))


    # In[21]:


    data = df.filter(['Close'])


    # In[22]:


    type(data)


    # In[23]:


    data.shape


    # In[24]:


    df.describe()
    df['Close'].isna().sum()


    # In[25]:


    close_data = data.dropna().values


    # In[26]:


    print(close_data.shape)
    type(close_data)


    # In[27]:


    training_len = int(len(close_data)*0.8)
    training_len


    # In[28]:


    scaler = MinMaxScaler(feature_range = (0,1))
    scaled_data = scaler.fit_transform(close_data)
    scaled_data


    # In[29]:


    train_data = scaled_data[:training_len,:]
    x_train = []
    y_train = []
    for i in range(60,len(train_data)):
        x_train.append(train_data[i-60:i,0])
        y_train.append(train_data[i,0])


    # In[30]:


    print(len(x_train),len(y_train))


    # In[31]:


    x_train,y_train = np.array(x_train),np.array(y_train)
    x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
    x_train.shape


    # In[15]:


    model = Sequential()
    model.add(LSTM(64,return_sequences = True, input_shape = (x_train.shape[1],1)))
    model.add(LSTM(50,return_sequences = False))
    #model.add(Dropout(0.2))
    model.add(Dense(32))
    #model.add(Dropout(0.2))
    model.add(Dense(1))


    # In[32]:


    model.compile(optimizer = 'Adam', loss = 'mean_squared_error')


    # In[33]:


    model.fit(x_train,y_train,batch_size=1,epochs=1)


    # In[34]:


    test_data = scaled_data[training_len-60:,:]
    x_test = []
    y_test = close_data[training_len:,:]
    for i in range(60,len(test_data)):
        x_test.append(test_data[i-60:i,0])


    # In[35]:


    x_test = np.array(x_test) #,np.array(y_test)
    x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))


    # In[36]:


    predictions = model.predict(x_test)
    preds = scaler.inverse_transform(predictions)


    # In[37]:


    preds.shape,y_test.shape


    # In[38]:


    rmse = np.sqrt(np.mean(preds-y_test)**2)
    rmse


    # In[39]:


    '''
    train = close_data[:training_len]
    test = close_data[training_len:]
    print(test.shape)

    test['Predictions'] = preds
    plt.figure(figsize=(16,8))
    plt.plot(train['Close'])
    plt.plot(test['Close','Predictions'])
    plt.legend('Train','Validate','Predictions',loc='lower right')
    plt.show()
    '''
    from tkinter import messagebox
    messagebox.showinfo("Result","Results are Ready")

    # In[40]:


    #date = df.iloc[training_len:,0:1]

    #dates = np.array(date)
    plt.figure(figsize=(16,8))
    #plt.plot(y_test1,linewidth=3)
    #plt.plot(pred1,linewidth=3)
    plt.plot(y_test, linewidth=3)
    plt.plot(preds,linewidth=3)
    plt.ylabel('BNF', fontsize =18)
    plt.legend(('Actual close price','Predicted close price','Actual open price','Actual open price'))
    plt.show()


    # In[ ]:


    pred1=preds
    y_test1=y_test


    # In[ ]:


    for i in range(0,len(y_test)):
     print(y_test[i],preds[i])


    # In[ ]:


    #y_test['Preds']=preds


    # In[ ]:


    print(df.iloc[training_len:,0:1])


    # In[ ]:


    fig, ax = plt.subplots()
    ax.plot('Date', 'Close', data=df)
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




