"""
Hello World for tkinter.
https://docs.python.org/3/library/tkinter.html 
"""
import tkinter as tk

class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.frame = tk.Frame(self.root)
        self.frame.pack() # geometry manager, can also use grid or place instead
        self.root.title("Hello World App")
        self.root.geometry("200x60")  # Set the window size
        self.root.resizable(False, False)  # Disable resizing
        self.root.iconname("hello_tk")  # Set the window title
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self.frame)
        self.hi_there["text"] = "Say Hi!"
        self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side="top")

        self.quit_button = tk.Button(self.frame, text="Quit", fg="red",
                                     command=self.root.destroy)
        self.quit_button.pack(side="bottom")

    def say_hi(self):
        print("Hi there, everyone!")

if __name__ == "__main__":
    app = Application()
    app.root.mainloop()