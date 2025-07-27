import tkinter as tk	
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar

import numpy as np

class Application:
	def __init__(self):
		self.root = tk.Tk()
		self.frame = tk.Frame(self.root)
		self.frame.pack(padx=15,pady=15)

		self.fig = Figure(figsize=(6, 4), dpi=100)
		self.ax = self.fig.add_subplot(111)
		x = np.linspace(0, 4)
		self.ax.plot(x, np.sin(x))
		self.canvas = FigureCanvas(self.fig, master=self.frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
		self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)
		self.mpl_toolbar.update()
		self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

if __name__ == '__main__':
    app = Application()
    app.root.mainloop()