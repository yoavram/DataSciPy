import tkinter as tk
from tkinter import filedialog, messagebox
import pygubu
from pathlib import Path

class NotepadApp:
    def __init__(self):
        self.master = tk.Tk()
        self.builder = pygubu.Builder()
        # Load the UI file, load main window, connect callbacks
        ui_path = Path(__file__).parent / 'notepad.ui'
        self.builder.add_from_file(ui_path)
        self.mainwindow = self.builder.get_object('mainwindow', self.master)
        self.builder.connect_callbacks(self)

        # Get widgets by id
        self.textEdit = self.builder.get_object('textEdit')
        self.filenameEdit = self.builder.get_object('filenameEdit')
        self.counterLabel = self.builder.get_object('counterLabel')

        # Bind word count update to text changes
        self.textEdit.bind('<<Modified>>', self.on_text_modified)
        self.update_word_count()

    def save(self):
        filename = self.filenameEdit.get()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.textEdit.get("1.0", tk.END))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def load(self):
        filename = self.filenameEdit.get()
        try:
            with open(filename, encoding='utf-8') as f:
                content = f.read()
            self.textEdit.delete("1.0", tk.END)
            self.textEdit.insert(tk.END, content)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def browse(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Text Files", "*.txt *.csv *.json"), ("All Files", "*.*")]
        )
        if filename:
            self.filenameEdit.delete(0, tk.END)
            self.filenameEdit.insert(0, filename)

    def update_word_count(self):
        text = self.textEdit.get("1.0", tk.END).strip()
        n_words = len(text.split())
        self.counterLabel.config(text=f"Words: {n_words}")

    def on_text_modified(self, event=None):
        self.update_word_count()
        self.textEdit.edit_modified(False) # finished handling the event


if __name__ == "__main__":
    app = NotepadApp()
    app.master.mainloop()