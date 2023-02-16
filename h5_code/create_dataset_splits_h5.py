import numpy as np
import h5py
import random


def write_batch_h5_file(dataset_file, split_name, sample_keys, resolution):
    batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
    labels = []
    batch_idx = 0
    num_of_batches = 0

    hf = h5py.File(dataset_file, 'r')

    file_name = split_name + '.h5'
    hf_split = h5py.File(file_name, 'a')

    for key in sample_keys:
    #     group = hf.get(key)
    #     x = np.array(group.get("x"), dtype=np.float32)
    #     y = np.array(group.get("y"), dtype=np.int8)
        with h5py.File('filename.h5', 'r') as f:
    # # Load the 'x' and 'y' datasets from the file
         x = np.array(f['group/x'], dtype=np.float32)
         y = np.array(f['group/y'], dtype=np.int8)
        batch[batch_idx, :, :, :, :] = x
        batch_idx += 1
        labels.append(y)

        if batch_idx == batch_size:
            print(f"Batch Num: {num_of_batches}")
            group = hf_split.create_group(str(num_of_batches))
            group.create_dataset("x", data=batch, compression="lzf")
            group.create_dataset("y", data=labels)

            batch = np.zeros((batch_size, 1, resolution, resolution, resolution), dtype=np.float32)
            batch_idx = 0
            num_of_batches += 1
            labels = []

    hf.close()
    hf_split.close()


def split_dataset(split, samples):
    train_idx = int(split["train"] * len(samples))
    val_idx = int((split["val"] * len(samples)) + train_idx)

    train_list = samples[:train_idx]
    val_list = samples[train_idx:val_idx]
    test_list = samples[val_idx:]

    return train_list, val_list, test_list


def read_voxel_from_h5(file):
    hf = h5py.File(file, 'r')
    keys = list(hf.keys())
    random.shuffle(keys)
    random.shuffle(keys)
    train_samples, val_samples, test_samples = split_dataset(dataset_split, keys)
    hf.close()

    return train_samples, val_samples, test_samples


def test_h5(split):
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


    hf.close()

    print(labels)


if __name__ == '__main__':
    batch_size = 40
    voxel_resolution = 64
    dataset_split = {"train": 0.7, "val": 0.15, "test": 0.15}

    train_samples, val_samples, test_samples = read_voxel_from_h5("voxel.h5")

    print("Train")
    write_batch_h5_file("voxels.h5", "train", train_samples, voxel_resolution)
    print("Validation")
    write_batch_h5_file("voxels.h5", "val", val_samples, voxel_resolution)
    print("Test")
    write_batch_h5_file("voxels.h5", "test", test_samples, voxel_resolution)

