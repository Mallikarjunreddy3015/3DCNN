import glob
import math
import numpy as np
import matplotlib.pyplot as plt
import h5py
import random
import utils.binvox_rw as binvox_rw
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go


def write_batches_for_split(split_name, sample_keys, resolution, norm):
    batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
    labels = []
    names = []
    batch_idx = 0
    num_of_batches = 0

    file_name = split_name + '.h5'
    hf = h5py.File(file_name, 'a')

    for key in sample_keys:
        x, y = read_voxel_from_binvox(key, norm)

        batch[batch_idx, :, :, :, :] = x
        batch_idx += 1
        labels.append(y)
        names.append(key)

        if batch_idx == batch_size:
            print(f"Batch Num: {num_of_batches}")
            group = hf.create_group(" " +str(num_of_batches))
            cad_names = np.array(names, dtype="S")
            group.create_dataset("names", data=cad_names, compression="lzf")
            group.create_dataset("x", data=batch, compression="lzf")
            group.create_dataset("y", data=labels)

            batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
            batch_idx = 0
            num_of_batches += 1
            labels = []
            names = []

    hf.close()


def read_h5(split):
    hf = h5py.File(split + ".h5", 'r')
    labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0,
              16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0}

    for key in list(hf.keys()):
        with h5py.File('filename.h5', 'r') as f:
    # # Load the 'x' and 'y' datasets from the file
         x = np.array(f['group/x'], dtype=np.float32)
         y = np.array(f['group/y'], dtype=np.int8)

        print(f"Group: {f}")
        print(f"X: {np.shape(x)}")
        print(f"Y: {y}")

        for label in y:
            labels[label] += 1

    hf.close()

    print(labels)


def split_dataset(split, samples):
    random.shuffle(samples)
    random.shuffle(samples)

    train_idx = int(math.ceil(split["train"] * len(samples)))
    val_idx = int(math.ceil((split["val"] * len(samples))) + train_idx)

    train_list = samples[:train_idx]
    val_list = samples[train_idx:val_idx]
    test_list = samples[val_idx:]

    return train_list, val_list, test_list


def read_voxel_from_binvox(filepath, normalize=True):
    with open(filepath, "rb") as f:
        model = binvox_rw.read_as_3d_array(f)
    voxel = model.data

    if normalize:
        voxel = zero_centering_norm(voxel)

    filename = filepath.split("\\")[-1]
    label = filename.split("_")[0]
    if  label.isdigit():
     voxel = np.array(voxel, dtype=np.float32)
     label = np.array(label, dtype=np.int8)
    


    

    return voxel, label


def zero_centering_norm(voxels):
    norm = (voxels - 0.5) * 2
    return norm


def display_voxel(voxels):
    fig = go.Figure(data=go.Volume(
        x=voxels[:, :, :, 0],
        y=voxels[:, :, :, 1],
        z=voxels[:, :, :, 2],
        value=voxels[:, :, :, 3],
        isomin=0,
        isomax=1,
        opacity=0.1,
        surface_count=21,
        ))
    fig.show()



if __name__ == '__main__':
    # Parameters to set
    main_dir = "C:/Users/reddy/Desktop/VISHALBTP/Dataset/"
    batch_size = 40
    voxel_resolution = 64
    normalize = True
    dataset_split = {"train": 0.7, "val": 0.15, "test": 0.15}

    
    list_of_files = glob.glob(main_dir + "*.binvox")
    x=1
    for f in list_of_files:
        x+=1
        print(x,f)
        
    train_samples, val_samples, test_samples = split_dataset(dataset_split, list_of_files)

    print("Train")
    write_batches_for_split("train", train_samples, voxel_resolution, normalize)
    print("Validation")
    write_batches_for_split("val", val_samples, voxel_resolution, normalize)
    print("Test")
    write_batches_for_split("test", test_samples, voxel_resolution, normalize)

