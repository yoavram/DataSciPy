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

#### Open the notebooks in VS Code

- Open `index.ipynb` in VS Code.
- When VS Code asks for a kernel, choose the `.venv` environment you created.

## The Keras backend

This course runs **Keras 3 on the JAX backend** and does not install TensorFlow.

Keras does not know that. Left to itself it defaults to the TensorFlow backend, and
because TensorFlow is not installed, `import keras` fails outright. So the backend has
to be selected, and the course folder does that for you: the **`.env`** file sets

```
KERAS_BACKEND=jax
```

and VS Code's Python extension loads it automatically for every notebook and terminal
in this folder. **If you followed the setup guide above, there is nothing to do.**

Two things to know in case you go off that path:

- **Do not delete or rename `.env`**, and keep the notebooks inside the course folder.
  A notebook moved elsewhere loses the setting.
- **Jupyter launched from a terminal does not read `.env`** (nor does `nbconvert`, or
  plain `python`). On that route, set the variable yourself before starting Jupyter:

  ```bash
  export KERAS_BACKEND=jax          # macOS / Linux
  $env:KERAS_BACKEND = "jax"        # Windows PowerShell
  ```

  [LOCAL_SETUP.md](LOCAL_SETUP.md) shows how to make it permanent for your user
  account instead.

### Troubleshooting

**`ModuleNotFoundError: No module named 'tensorflow'`**

This is the symptom of the backend not being set — it appears the moment a notebook
runs `import keras`, and it means Keras tried to start its default TensorFlow backend.
Nothing is broken in your installation and you do **not** need to install TensorFlow;
installing it would give you two deep learning frameworks and a slower, more confusing
setup.

Check what Keras thinks it is using:

```python
import os
print(os.environ.get('KERAS_BACKEND'))   # should print: jax
```

If that prints `None`, the `.env` file is not reaching your kernel. Either you are not
in VS Code, the course folder is not the folder you opened, `.env` is missing, or the
**Python** extension is not installed. Fix whichever applies, then **restart the
kernel** — changing `.env` has no effect on a kernel that is already running.

As an immediate escape hatch, add this to the very top of the notebook, *above* the
cell that imports Keras, and restart the kernel:

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
```

**A note for the curious:** Keras picks its backend from the `KERAS_BACKEND`
environment variable first, then from a `keras.json` file in your home directory
(`~/.keras/keras.json`), and defaults to TensorFlow if neither says otherwise. That
home-directory file is per-machine, not part of this course folder — which is why a
computer that has run Keras before may work while a freshly set up one does not.
