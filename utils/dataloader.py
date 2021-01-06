import h5py
import tensorflow as tf
import numpy as np


def dataloader_h5(file_path):
    hf = h5py.File(file_path, 'r')

    for key in list(hf.keys()):
        group = hf.get(key)
        x = tf.Variable(np.array(group.get("x"), dtype=np.float32), dtype=tf.float32, name="x")
        y = np.array(group.get("y"), dtype=np.int8)

        yield x, y

    hf.close()