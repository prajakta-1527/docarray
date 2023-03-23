from typing import Any

import numpy as np

try:
    import torch  # noqa: F401
except ImportError:
    torch_imported = False
else:
    torch_imported = True


try:
    import tensorflow as tf  # type: ignore # noqa: F401
except (ImportError, TypeError):
    tf_imported = False
else:
    tf_imported = True


def get_hnswlib():
    try:
        import hnswlib

        return hnswlib
    except (ImportError, ModuleNotFoundError):
        raise RuntimeError(
            "Hnswlib is not installed. To install all the necessary libraries to use the hnsw backend, "
            "please do: `pip install docarray[hnswlib]`."
        )


def is_torch_available():
    return torch_imported


def is_tf_available():
    return tf_imported


def is_np_int(item: Any) -> bool:
    dtype = getattr(item, 'dtype', None)
    ndim = getattr(item, 'ndim', None)
    if dtype is not None and ndim is not None:
        try:
            return ndim == 0 and np.issubdtype(dtype, np.integer)
        except TypeError:
            return False
    return False  # this is unreachable, but mypy wants it


def is_notebook() -> bool:
    """
    Check if we're running in a Jupyter notebook, using magic command
    `get_ipython` that only available in Jupyter.

    :return: True if run in a Jupyter notebook else False.
    """

    try:
        shell = get_ipython().__class__.__name__  # type: ignore
    except NameError:
        return False

    if shell == 'ZMQInteractiveShell':
        return True

    elif shell == 'Shell':
        return True

    elif shell == 'TerminalInteractiveShell':
        return False

    else:
        return False
