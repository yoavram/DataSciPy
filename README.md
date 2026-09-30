# Introduction to Deep Learning
## Yoav Ram

Go to the [index notebook](index.ipynb) to view the table of contents.

## Setup guide

This guide gets you ready to run the course Jupyter notebooks in VS Code using the official Jupyter extension.

#### Install VS Code

- Download and install from <https://code.visualstudio.com/>.

#### Download the course materials

- Direct download (recommended): [ZIP file](https://github.com/yoavram/DataSciPy/archive/refs/heads/DL2026.zip).
- Unzip and note the folder path.
- If you prefer `git`, see <https://github.com/yoavram/DataSciPy/tree/DL2026>.

#### Open the course folder in VS Code

- Start VS Code.
- Choose **File -> Open Folder...**
- Select the course folder you just unzipped.

#### Install VS Code extensions

- Open **Extensions** in VS Code and install:
  - **Python** (by Microsoft): <https://code.visualstudio.com/docs/python/python-quick-start>
  - **Jupyter** (by Microsoft): <https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter>
  - Optional: **GitHub Copilot**: <https://code.visualstudio.com/docs/copilot/setup>

#### Create a Python environment

- In VS Code, open the Command Palette (Ctrl+Shift+P) and run **Python: Create Environment**.
- Choose *venv*.
- Choose a recent Python version, preferably Python 3.12 or 3.13.
- When prompted for virtual environment name, stick with `.venv`.
- Choose *Install project dependencies* 
- Choose the `requirements.txt` file from the local folder `DataSciPy/requirements.txt`.

VS Code will now create a virtual environment with course dependencies, which can take a few minutes.

#### Install KerasHub

One package cannot go in `requirements.txt`. The CLIP section of
[Open-set metric learning](sessions/metric_learning.ipynb) needs
[KerasHub](https://keras.io/keras_hub/), and `keras-hub` declares `tensorflow-text` as a
dependency — which would install TensorFlow, which this course deliberately does not use.
So install it *without* its dependency tree. Open a terminal in VS Code
(**Terminal -> New Terminal**, with the `.venv` environment active) and run:

```bash
python -m pip install --no-deps keras-hub==0.32.0
```

Everything it actually needs is already in `requirements.txt`. Skip this if you are not
doing that session; nothing else in the course imports `keras_hub`.

#### Set up a Kaggle token (only for the re-identification session)

[Animal re-identification](sessions/reid.ipynb) uses the SeaTurtleIDHeads archive, which is
distributed through Kaggle. `wildlife-datasets` is already in `requirements.txt`, but the
download needs an API token:

1. Sign in at <https://www.kaggle.com>, open **Settings**, and choose **Create New Token**.
   This downloads `kaggle.json`.
2. Move it to `~/.kaggle/kaggle.json` (on Windows, `C:\Users\<you>\.kaggle\kaggle.json`).
3. Visit <https://www.kaggle.com/datasets/wildlifedatasets/seaturtleidheads> once and accept
   the dataset's terms.

Then fetch it with `python download_data.py turtles` (about 425 MB). Skip this if you are not
doing that session.

#### Open the notebooks in VS Code

- Open `index.ipynb` in VS Code.
- When VS Code asks for a kernel, choose the `.venv` environment you created.

## Setting up from a terminal (another machine, or without VS Code)

The steps above are the supported route. To build the same environment from a terminal,
with Python 3.12 and the course folder as the working directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install --no-deps keras-hub==0.32.0   # see "Install KerasHub" above
export KERAS_BACKEND=jax             # VS Code reads .env; a terminal does not
python -c "import keras, jax; print(keras.__version__, keras.backend.backend(), jax.default_backend())"
python download_data.py              # the trained checkpoints; none are committed
```

`requirements.txt` pins every package it lists to an exact version, but not the packages
those depend on, so two installs can still differ in those. To reproduce an environment
exactly, freeze a working one and install from the result on the other machine:

```bash
python -m pip freeze > requirements.lock     # on the working machine
python -m pip install -r requirements.lock   # on the other one
```

On a machine with an NVIDIA GPU, install the CUDA build of JAX
(`python -m pip install "jax[cuda12]==0.11.1"`, matching your driver) after the step
that installs `requirements.txt`. The plain `jax` line installs the CPU build.

## Troubleshooting

If a notebook will not run — in particular if you see
`ModuleNotFoundError: No module named 'tensorflow'` — see
**[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**. It explains how this course selects the
Keras backend, what to do when that goes wrong, and what changes if you work outside
VS Code.

If you followed the steps above, you should not need it.
