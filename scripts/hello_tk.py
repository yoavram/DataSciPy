"""
Hello World for tkinter.
https://docs.python.org/3/library/tkinter.html 
"""
import tkinter as tk

class Application:
    def __init__(self):
        self.master = tk.Tk()
        self.frame = tk.Frame(self.master)
        self.frame.grid() # geometry manager, can also use grid or place instead
        self.master.title("Hello World App")
        self.master.geometry("200x60")  # Set the window size
        self.master.resizable(False, False)  # Disable resizing

        self.hi_btn = tk.Button(
            self.frame, 
            text="Say Hi!",
            command=self.on_click
        )
        self.hi_btn.grid(row=0, column=0)

        self.label = tk.Label(self.frame, text='')
        self.label.grid(row=0, column=1)

        self.quit_btn = tk.Button(
            self.frame, 
            text="Quit", 
            fg="red",
            command=self.master.destroy
        )
        self.quit_btn.grid(row=1, column=0)

    def on_click(self):
        self.label.config(text="Hi!")

if __name__ == "__main__":
    app = Application()
    app.master.mainloop()