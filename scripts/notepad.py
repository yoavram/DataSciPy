import sys
import threading
import time
from PySide import QtGui
from notepad_design import Ui_MainWindow

class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # connect buttons to callbacks
        self.saveButton.clicked.connect(self.save)
        self.loadButton.clicked.connect(self.load)
        self.browseButton.clicked.connect(self.browse)

        # create word count update worker thread
        def word_counter_update_task():
            while True:
                self.update_word_count()
                time.sleep(1)
        self.word_count_updater = threading.Thread(target=word_counter_update_task, name='word_count_updater')
        self.word_count_updater.start()

        self.show()

    def error_dialog(self, error):
        msg = QtGui.QMessageBox(self)
        msg.setText(str(error))
        msg.show()

    def save(self):
        text = self.textEdit.toPlainText()
        filename = self.filenameEdit.text()
        try:
            with open(filename, 'w') as f:
                print(text, file=f)
        except Exception as e:
                self.error_dialog(e)

    def load(self):
        filename = self.filenameEdit.text()
        try:
            with open(filename) as f:
                text = f.read()
        except Exception as e:
            self.error_dialog(e)
        self.textEdit.clear()
        self.textEdit.appendPlainText(text)

    def browse(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(
            self,
            "Open Text File",
            ".",
            "Text Files (*.txt *.csv *.json)"
        )
        self.filenameEdit.setText(filename)

    def update_word_count(self):
        text = self.textEdit.toPlainText()
        words = text.split()
        word_count = len(words)
        self.counterLabel.setText('{:d}'.format(word_count))


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    form = MainWindow()
    form.show()
    sys.exit(app.exec_())