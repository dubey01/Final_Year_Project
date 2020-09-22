import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb
from tkinter import *
from tkinter.ttk import *
import LSTM9
from LSTM9 import CSV_file
import sys
import list_L
import search
import accinv
import PRC




LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.iconbitmap(self, default="")
        tk.Tk.wm_title(self, "Prediciton")
        
        
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

    def answer():
        mb.showerror("Answer", "Sell the stock")

    #progress = Progressbar(container, orient = HORIZONTAL, 
              #length = 100, mode = 'determinate') 
  
# Function responsible for the updation 
# of the progress bar value 
    def bar():
        if(LSTM9.CSV_file == ""):
            mb.showerror("Error", "Select stocks to sell")
            return
            
        root = Tk() 
        progress = Progressbar(root, orient = HORIZONTAL, length = 100, mode = 'determinate') 
        import time
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

        if(LSTM9.CSV_file == ""):
            mb.showerror("Error", "Select stocks to sell")
        else:
            mb.showinfo("Sell", LSTM9.CSV_file+" stocks")
            from accinv import tempList
            while(LSTM9.CSV_file in tempList):
                tempList.remove(LSTM9.CSV_file)

            mb.showinfo("Successful", LSTM9.CSV_file+" Stock Sold")

            with open('inventory', 'w') as f:
                for item in tempList:
                    f.write("%s\n" % item)

    def BUY():
        if(LSTM9.CSV_file == ""):
            mb.showerror("Error", "Select stocks to buy")
            return
        from accinv import tempList
        #import accinv
        file = open("inventory", "a+")
        file.write(LSTM9.CSV_file + "\r\n")
        file.close()
        accinv.tempList.append(LSTM9.CSV_file)
        mb.showinfo("Successful", LSTM9.CSV_file+" Stock Bought")
        #accinv.startacc()
        #progress.pack(pady = 10)     

    #def callback():
     #   if mb.askyesno('Verify', 'Really quit?'):
      #      mb.showwarning('Yes', 'Not yet implemented')
      #  else:
       #     mb.showinfo('No', 'Quit has been cancelled')

       
class StartPage(tk.Frame):


        

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Welcome To ADA Digital Locker and Stock Prediciton", font=("Calibri", 25), height="2", width="300")
        label.pack(pady=10,padx=10)
        #Label(text="Welcome", bg="gray", width="300", height="2", font=("Calibri", 13)).pack()
        #Label(text="").pack()
        user = StringVar()
        E1 = Entry(self, textvariable=user)
        E1.place(x=1000,y=100)
        user.set(E1.get())
        Button(self, text="Search", width=10, command =lambda: search.search_B(user)).place(x=1072,y=125)


        #Button(self, text="Search", width=10, height=1, command = search.search_B, user)).pack()
        button1 = ttk.Button(self, text="Stock List",
                            command=list_L.start)
        button1.pack()
        button0 = ttk.Button(self, text="Train",
                            command=LSTM9.neural_N)
        button0.pack()
        
        button = ttk.Button(self, text="My Stocks",
                            command=accinv.startacc)
        button.pack()

        button2 = ttk.Button(self, text="Feat Imp",
                            command=PRC.feat_imp)
        button2.pack()

        button3 = ttk.Button(self, text="Sell/Buy",
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
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Sell",
                            command=SeaofBTCapp.bar)
        button1.pack(side=tk.TOP)

        button2 = ttk.Button(self, text="Buy",
                            command=SeaofBTCapp.BUY)
        button2.pack(side=tk.TOP)


        f = Figure(figsize=(5,5), dpi=100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8],[5,6,1,3,8,9,3,5])

        

        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        

app = SeaofBTCapp()
app.mainloop()
