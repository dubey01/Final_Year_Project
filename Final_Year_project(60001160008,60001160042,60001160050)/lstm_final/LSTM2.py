#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CSV_file = ""

# In[2]:

def neural_N():
    
#df_ge=pd.read_csv('NSE-TATAGLOBAL.csv')
#print("checking if any null values are present\n", df_ge.isna().sum())


# In[3]:


# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

# Importing the training set
    #CSV_file  = 'NSE-TATAGLOBAL'
    df=pd.read_csv(CSV_file+'.csv')
    data_split_ratio=0.5
    dataset_train = df[:round(len(df)*data_split_ratio)]
    training_set = dataset_train.iloc[:, 1:2].values

    # Feature Scaling
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # Creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []
    for i in range(60, len(dataset_train)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshaping
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



    # Part 2 - Building the RNN

    # Importing the Keras libraries and packages
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from keras.layers import Dropout

    # Initialising the RNN
    regressor = Sequential()

    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))


    # Adding the output layer
    regressor.add(Dense(units = 1))

    # Compiling the RNN
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    regressor.fit(X_train, y_train, epochs = 1, batch_size = 32)



    # Part 3 - Making the predictions and visualising the results

    # Getting the real stock price of 2017
    dataset_test = df[round(len(df)*data_split_ratio):]
    real_stock_price = dataset_test.iloc[:, 1:2].values

    # Getting the predicted stock price
    dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(60, len(dataset_test)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # Visualising the results

    f = plt.figure(figsize=(20,10))
    plt.plot(real_stock_price, color = 'green', label = 'Real Stock Price')
    plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
    plt.title(' Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


    # In[16]:


    acc=(sum(predicted_stock_price)/sum(real_stock_price[0:len(real_stock_price)-60])*100)
    acc=100-abs(acc-100)-2
    print(acc)


    # In[30]:


    print(len(predicted_stock_price),len(real_stock_price[0:len(real_stock_price)-60]))


    # In[4]:


    X_test


    # In[ ]:





    # In[ ]:





    # In[6]:



    df=pd.read_csv(CSV_file+'.csv')
    print(len(df))
    dataset_train = df[:round(len(df)/2)]


# In[14]:



'''

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import *
from tkinter.ttk import *


LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="")
        tk.Tk.wm_title(self, "Predicted Results")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, PageOne, PageTwo, PageThree):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

    def bar():
        root = Tk()
        progress = Progressbar(root, orient = HORIZONTAL, length = 100, mode = 'determinate') 
        import time
       #root.progress.pack(anchor=Tkinter.CENTER)
        progress.pack(pady = 10)
        progress['value'] = 20
        root.update_idletasks() 
    #time.sleep(1) 
  
        progress['value'] = 40
        root.update_idletasks() 
        time.sleep(1) 
  
        progress['value'] = 50
        root.update_idletasks() 
    #time.sleep(1) 
  
        progress['value'] = 60
        root.update_idletasks() 
        time.sleep(1) 
  
        progress['value'] = 80
        root.update_idletasks() 
        time.sleep(1) 
        progress['value'] = 100

        root.destroy()

        mb.showerror("Success", "Transaction Successful")
    #def answer():
     #   mb.showerror("Answer", "Sorry, no answer available")

    #def callback():
     #   if mb.askyesno('Verify', 'Really quit?'):
      #      mb.showwarning('Yes', 'Not yet implemented')
      #  else:
       #     mb.showinfo('No', 'Quit has been cancelled')

       
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="History",
                            command=lambda: controller.show_frame(PageOne))
        button.pack()

        button2 = ttk.Button(self, text="Summary",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()

        button3 = ttk.Button(self, text="Results",
                            command=lambda: controller.show_frame(PageThree))
        button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page Two",
                            command=lambda: controller.show_frame(PageTwo))
        button2.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Page Two!!!", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        button2 = ttk.Button(self, text="Page One",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()


class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Forecast Representation", font=LARGE_FONT)
        label.pack(pady=5,padx=10)

        button1 = ttk.Button(self, text="Sell",
                            command=SeaofBTCapp.bar)
        button1.pack(side=tk.TOP)

        button2 = ttk.Button(self, text="Buy",
                            command=SeaofBTCapp.bar)
        button2.pack(side=tk.TOP)


        #f = Figure(figsize=(5,5), dpi=100)
        #a = f.add_subplot(111)
        #a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        f = plt.figure(figsize=(20,10))
    
        plt.title(' Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')

        plt.plot(real_stock_price, color = 'green', label = 'Real Stock Price')
        plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Stock Price')
        plt.legend()
        #plt.show()

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        

app = SeaofBTCapp()
app.mainloop()


# In[ ]:
'''



