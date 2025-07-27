try:
    import tkinter as tk  # for python 3
except ImportError:
    import Tkinter as tk  # for python 2
import pygubu


class Application:
    def __init__(self):
        self.master = master = tk.Tk()
        #1: Create a builder
        self.builder = builder = pygubu.Builder()

        #2: Load an ui file
        builder.add_from_file('hello_pygubu.ui')

        #3: Create the widget using a master as parent
        self.mainwindow = builder.get_object('mainwindow', master)
        self.lbl = builder.get_object('lbl', master)

        #3: Connect callbacks
        builder.connect_callbacks(self)

    def click(self):
        self.lbl['text'] = "Clicked"

if __name__ == '__main__':
    app = Application()
    app.master.mainloop()