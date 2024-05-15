# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import cloudpickle


def read(name, opener=open):
    """Load file contents as a bytestring."""
    with opener(name, "rb") as f:
        return f.read()


loads = cloudpickle.loads
dumps = cloudpickle.dumps


def load(name, opener=open):
    """Load a cloudpickled object from the named file."""
    with opener(name, "rb") as f:
        return cloudpickle.load(f)


def dump(obj, name, opener=open):
    """Dump an object to the named file using cloudpickle."""
    with opener(name, "wb") as f:
        cloudpickle.dump(obj, f)
