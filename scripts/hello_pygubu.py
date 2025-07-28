from pathlib import Path
import tkinter as tk
import pygubu

class Application:
    def __init__(self):
        self.master = tk.Tk()
        self.builder = pygubu.Builder()
        # Load the UI file
        app_folder = Path(__file__).parent
        ui_path = app_folder / 'hello_pygubu.ui'
        self.builder.add_from_file(ui_path)
        # Create the main window - must be done after loading the UI
        self.mainwindow = self.builder.get_object('mainwindow', self.master)        
        # Connect callbacks
        self.builder.connect_callbacks(self)

    def click(self):
        label = self.builder.get_object('lbl')
        label.config(text="Clicked")

if __name__ == '__main__':
    app = Application()
    app.master.mainloop()