# Troubleshooting

Setup instructions are in [README.md](README.md); the table of contents is in the
[index notebook](index.ipynb).

## The Keras backend

This course runs **Keras 3 on the JAX backend** and does not install TensorFlow.

Keras does not know that. Left to itself it defaults to the TensorFlow backend, and
because TensorFlow is not installed, `import keras` fails outright. So the backend has
to be selected, and the course folder does that for you: the **`.env`** file sets

```
KERAS_BACKEND=jax
```

and VS Code's Python extension loads it automatically for every notebook and terminal in
this folder. **If you followed the setup guide in the README, there is nothing to do** —
the rest of this section is only for when something has gone wrong.

### `ModuleNotFoundError: No module named 'tensorflow'`

This is the symptom of the backend not being set. It appears the moment a notebook runs
`import keras`, and it means Keras tried to start its default TensorFlow backend.

Nothing is broken in your installation, and you do **not** need to install TensorFlow.
Installing it would give you two deep learning frameworks, a slower setup, and a more
confusing one.

Check what Keras is being told to use:

```python
import os
print(os.environ.get('KERAS_BACKEND'))   # should print: jax
```

If that prints `None`, the `.env` file is not reaching your kernel. The usual causes, in
rough order of likelihood:

- you are not running in VS Code (see below);
- the folder you opened in VS Code is not the course folder, so `.env` is not in scope;
- the **Python** extension is not installed, so nothing loads `.env`;
- `.env` was deleted, renamed, or left behind when the notebook was moved.

Fix whichever applies, then **restart the kernel** — changing `.env` has no effect on a
kernel that is already running.

As an immediate escape hatch, add this at the very top of the notebook, *above* the cell
that imports Keras, and restart the kernel:

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'
```

### If you are not using VS Code

**Jupyter launched from a terminal does not read `.env`** — nor does `nbconvert`, nor a
bare `python`. On that route, set the variable yourself before starting Jupyter:

```bash
export KERAS_BACKEND=jax          # macOS / Linux
$env:KERAS_BACKEND = "jax"        # Windows PowerShell
```

To make it permanent for your user account instead of setting it in every shell, create
`~/.keras/keras.json` containing:

```json
{"floatx": "float32", "epsilon": 1e-07, "backend": "jax", "image_data_format": "channels_last"}
```

Also keep the notebooks **inside the course folder**. A notebook moved elsewhere loses
the setting, and its relative paths to `../data/` stop resolving too.

### How Keras actually picks its backend

For the curious, the order of precedence is:

1. the `KERAS_BACKEND` environment variable;
2. a `keras.json` file in your home directory (`~/.keras/keras.json`);
3. otherwise TensorFlow, which is why an unset backend fails here.

That home-directory file is per-machine and is not part of this course folder — which is
why a computer that has run Keras before may work while a freshly set up one does not.
