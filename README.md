# Deep Learning with Python workshop
## Yoav Ram

Go to the [index notebook](index.ipynb) to view the table of contents.

## Setup instructions

1. If you don't have Python, download and install from [Miniforge](https://conda-forge.org/download/), a Python installer that also includes conda, mamba
1. Download the workshop [ZIP file](https://github.com/yoavram/DataSciPy/archive/refs/heads/master.zip).
1. Unzip the ZIP; make note of the unzipped folder location.
1. Start the terminal or command line application.
1. Install required packages into the base environment: `mamba env update -f environment.yml`.

If using *Jupyter Lab*:
1. Change folder to the workshop folder (`cd <unzipped workshop folder>`)
1. Run Jupyter lab (`jupyter lab`) 
6. In the new browser window that just opened, choose [`index.ipynb`](index.ipynb).

If using *VS Code* (requires additional install from [here](https://code.visualstudio.com/download)):
1. Open VS Code and open the workshop folder.

### Notes to experienced users
1. Instead of downloading the ZIP, you can clone the repo: `git clone https://github.com/yoavram/DataSciPy.git`
1. Instead of updating the base environment, you can create a new environment: `mamba env create -f environment.yml -n DataSciPy`.
