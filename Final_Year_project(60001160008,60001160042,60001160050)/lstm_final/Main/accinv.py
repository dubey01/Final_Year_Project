import tkinter as tk
from tkinter import ttk
from functools import partial

tempList = []
f = open("inventory")
tempList = f.read().splitlines()
f.close()

while("" in tempList) : 
    tempList.remove("")

        
def showacc(listBox):


    #tempList.sort(key=lambda e: e[1], reverse=True)

    for i, (name) in enumerate(tempList, start=1):
        listBox.insert("", "end", values=(i, name))

def startacc():
    
    scores = tk.Tk()
    label = tk.Label(scores, text="Procured Stocks", font=("Arial",30)).grid(row=0, columnspan=3)
    # create Treeview with 3 columns
    cols = ('Serial', 'Stock Name')
    listBox = ttk.Treeview(scores, columns=cols, show='headings')
    # set column headings
    for col in cols:
        listBox.heading(col, text=col)    
    listBox.grid(row=1, column=0, columnspan=5)

    showScores = tk.Button(scores, text="Show Stocks", width=15, command=partial(showacc, listBox)).grid(row=4, column=0)
    closeButton = tk.Button(scores, text="Close", width=15, command=exit).grid(row=4, column=1)

    scores.mainloop()
