import tkinter as tk	
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigureCanvas
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar

import numpy as np

class Application:
	def __init__(self):
		self.master = tk.Tk()
		self.frame = tk.Frame(self.master)
		self.frame.pack()

		self.fig = plt.figure(figsize=(6, 4), dpi=100)
		self.ax = self.fig.add_subplot(111)
		x = np.linspace(0, 4)
		self.line = self.ax.plot(x, np.sin(x))[0]
		self.canvas = FigureCanvas(self.fig, master=self.frame)
		self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
		self.mpl_toolbar = NavigationToolbar(self.canvas, self.frame)
		self.mpl_toolbar.update()

		self.btn = tk.Button(
			self.master, 
			text="Hide", 
			command=self.on_click
		)
		self.btn.pack(side=tk.BOTTOM)

	def on_click(self):
		if self.line.get_visible():
			self.line.set_visible(False)
			self.canvas.draw()
			self.btn['text'] = 'Show'
		else:
			self.line.set_visible(True)
			self.canvas.draw()
			self.btn['text'] = 'Hide'


if __name__ == '__main__':
    app = Application()
    app.master.mainloop()