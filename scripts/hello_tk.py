"""Hello World for tkinter.
https://docs.python.org/3/library/tkinter.html 
"""
import tkinter as tk

class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root)
        self.frame.pack()
        self.createWidgets()

    def createWidgets(self):
        self.hi_there = tk.Button(self.frame)
        self.hi_there["text"] = "Hello World\n(click me)"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self.frame, text="Quit", fg="red",
                              command=self.root.destroy)
        self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

app = Application()
app.root.mainloop()