from tkinter import *
from functools import partial
import os
from LSTM9 import CSV_file
import LSTM9
from tkinter import messagebox

    
def search_B(k):
    
    flag = 0
    for dirpath, dirnames, files in os.walk('.'):
        for file in files:
            name = file.split('.')[0]
            if(k.get().lower() == name.lower()):
                LSTM9.CSV_file = name
                messagebox.showinfo("Success","Search Successful")
                flag = 1
                break
        if(flag == 1):
            break
        else:
            messagebox.showerror("Failure", "Search Unsuccessful")
            break
           
#def win():
'''    
top = Tk()
userlok = StringVar()
L1 = Label(top, text="User Name")
L1.pack( side = LEFT)
E1 = Entry(top, bd =5, textvariable=userlok)
E1.pack(side = RIGHT)
    #print(E1.get())

Button(top, text="Search", width=10, height=1, command = partial(search_B, userlok)).pack()
Button(top, text="Train",command=LSTM9.neural_N).pack()
top.mainloop()
'''

