import h5py
import tensorflow as tf
import numpy as np
import utils.binvox_rw as binvox_rw
from create_dataset_splits import zero_centering_norm


# def dataloader_h5(file_path):
#     hf = h5py.File(file_path, 'r')

#     for key in list(hf.keys()):
#         group = hf.get(key)
#         x = tf.convert_to_tensor(np.array(group.get("x"), dtype=np.float32))
#         x = tf.Variable(x, dtype=tf.float32, name="x")
 
#         y = np.array(group["y"], dtype=np.int8)

#         yield x, y

#     hf.close()
def dataloader_h5(file_path):
    hf = h5py.File(file_path, 'r')

    for key in hf.keys():
        group = hf.get(key)
        if group is not None and isinstance(group, h5py.Group):
            x = tf.convert_to_tensor(np.array(group["x"], dtype=np.float32))
            x = tf.Variable(x, dtype=tf.float32, name="x")
            y = np.array(group["y"], dtype=np.int8)
            yield x, y

    hf.close()


def read_voxel_from_binvox(filepath, normalize=True):
    with open(filepath, "rb") as f:
        model = binvox_rw.read_as_3d_array(f)
    voxel = model.data

    if normalize:
        voxel = zero_centering_norm(voxel)

    filename = filepath.split("/")[-1]
    #label = int(filename.split("-")[0])

    # voxel = tf.Variable(np.array(voxel, dtype=np.float32), dtype=tf.float32, name="x")
    #label = np.array(label, dtype=np.int8)
    voxel = tf.convert_to_tensor(np.array(voxel, dtype=np.float32))
    voxel = tf.Variable(voxel, dtype=tf.float32, name="x")


    return voxel