# Data Science with Python workshop
## Setup instructions

- Install [Anaconda with Python 3](http://anaconda.com/download/)
- Download the workshop files from [GitHub](https://github.com/yoavram/DataSciPy/tree/amat2019) by clicking "Clone or download->Download ZIP" (you can `git clone` if you know how to).
- Unzip the ZIP; make note of the unzipped folder location.
- Start the *Anaconda Prompt* terminal application.
- Run Jupyter from *Anaconda Prompt* after changing folder to the workshop folder:
```sh
cd <unzipped workshop folder>
jupyter lab
```
- In the new browser window that just opened, choose [`index.ipynb`](index.ipynb).
- Download the required data files for Day 3; open *Anaconda Prompt*/*Terminal*, activate the environment (on Mac and Linux, start the first line with `source`), and run the download script:
```sh
> cd <unzipped workshop folder>
> python download_data.py
```
