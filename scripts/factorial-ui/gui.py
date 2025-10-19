#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the MIT license:
# http://www.opensource.org/licenses/MIT-license
# Copyright (c) 2016, Yoav Ram <yoav@yoavram.com>
from tkinter import *
from tkinter import ttk
import pygubu


def factorial(n):
	'''Calculate the factorial of n using recursion.
	'''
	if n < 1:
		return 1
	else:
		return n * factorial(n-1)


class FactorialGUI:
	def __init__(self, master=None):
		self.master = master 
		self.builder = pygubu.Builder()
		self.builder.add_from_file('gui.ui')
		self.mainwindow = self.builder.get_object('mainwindow', master)

		self.input_var = StringVar()
		self.output_var = StringVar()

		input_entry = self.builder.get_object('inputEntry', master)
		input_entry['textvariable'] = self.input_var
		
		output_label = self.builder.get_object('outputLabel', master)
		output_label['textvariable'] = self.output_var
		
		calculate_button = self.builder.get_object('calculateButton', master)
		calculate_button['command'] = self.calculate

		self.master.bind('<Return>', self.calculate)
		input_entry.focus()

	def calculate(self, *args):
		try:
			n = int(self.input_var.get())
			self.output_var.set(factorial(n))
		except ValueError as e:
			tkinter.messagebox.showinfo("Error: {}".format(e))
			pass


if __name__ == '__main__':
	app = Tk()
	app.title("Factorial")
	window = FactorialGUI(app)
	app.mainloop()

