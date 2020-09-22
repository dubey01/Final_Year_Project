import tkinter as tk
from tkinter import ttk
from functools import partial

def show(listBox):

    tempList = [['ACC Ltd', 'ACC'],	
['Adani Ports And Special Economic Zone Ltd', 'ADANIPORTS'],	
['Ambuja Cements Ltd', 'AMBUJACEM'],			
['Axis Bank Ltd', 'AXISBANK'],	
['Bajaj Auto Ltd', 'BAJAJ-AUTO'],	
['Bank Of Baroda', 'BANKBARODA'],	
['Bharat Heavy Electricals Ltd', 'BHEL'],	
['Bharat Petroleum Corporation Ltd', 'BPCL'],	
['Bharti Airtel Ltd', 'BHARTIARTL']]
    tempList.sort(key=lambda e: e[1], reverse=True)

    for i, (name, symbol) in enumerate(tempList, start=1):
        listBox.insert("", "end", values=(i, name, symbol))

def start():
    
    scores = tk.Tk() 
    label = tk.Label(scores, text="Available Stocks", font=("Arial",30)).grid(row=0, columnspan=3)
    # create Treeview with 3 columns
    cols = ('Serial', 'Stock Name', 'Stock Symbol')
    listBox = ttk.Treeview(scores, columns=cols, show='headings')
    # set column headings
    for col in cols:
        listBox.heading(col, text=col)    
    listBox.grid(row=1, column=0, columnspan=5)

    showScores = tk.Button(scores, text="Show Stocks", width=15, command=partial(show, listBox)).grid(row=4, column=0)
    closeButton = tk.Button(scores, text="Close", width=15, command=exit).grid(row=4, column=1)

    scores.mainloop()
