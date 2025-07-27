import tkinter # in Python 2 use import Tkinter as tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2TkAgg as NavigationToolbar

import numpy as np

class MainWindow:
	def __init__(self, master=None):
		self.frame = tkinter.Frame(master)
		self.frame.pack(padx=15,pady=15)

		self.fig = matplotlib.figure.Figure(figsize=(6, 4), dpi=100)
		self.ax = self.fig.add_subplot(111)
		self.ax.plot(np.linspace(0, 4), np.sin(np.linspace(0, 4)))
		self.canvas = FigureCanvas(self.fig, master=self.frame)
		self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

		self.mpl_toolbar = NavigationToolbar(self.canvas, master)
		self.mpl_toolbar.update()
		self.canvas._tkcanvas.pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)


if __name__ == '__main__':
	app = tkinter.Tk()
	window = MainWindow(app)
	app.mainloop()