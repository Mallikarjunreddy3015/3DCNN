import tensorflow as tf
from skimage.segmentation import watershed
from scipy import ndimage as ndi
from network import FeatureNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import utils.binvox_rw as binvox_rw
from skimage import measure
from mpl_toolkits.mplot3d import Axes3D


def get_seg_samples(labels):

    samples = np.zeros((0, labels.shape[0], labels.shape[1], labels.shape[2]))

    for i in range(1, np.max(labels.astype(int)) + 1):
        idx = np.where(labels == i)

        if len(idx[0]) == 0:
            continue

        cursample = np.ones(labels.shape)
        cursample[idx] = 0
        cursample = np.expand_dims(cursample, axis=0)
        samples = np.append(samples, cursample, axis=0)

    return samples


def decomp_and_segment(sample):
    """Ref: https://github.com/PeizhiShi/MsvNet"""
    blobs = ~sample
    final_labels = np.zeros(blobs.shape)
    all_labels = measure.label(blobs)
    display_features(all_labels)  # Display connected component
    all_labels = np.array(all_labels)
    for i in range(1, np.max(all_labels) + 1):
        mk = (all_labels == i)
        distance = ndi.distance_transform_edt(mk)
        distance_arr = np.array(distance)
        labels = watershed(-distance_arr)

        max_val = np.max(final_labels) + 1
        idx = np.where(mk)

        final_labels[idx] += (labels[idx] + max_val)

    # display_features(final_labels) # Display watershed
    results = get_seg_samples(final_labels)

    return results


def get_bounding_box(voxel):
    a = np.where(voxel != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(
        a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
    return bbox




def zero_centering_norm(voxels):
    norm = (voxels - 0.5) * 2
    return norm


def load_voxel(file_path):
    hf = h5py.File(file_path, 'r')
    x = None

    for key in list(hf.keys()):
        with h5py.File('filename.h5', 'r') as f:
    # # Load the 'x' and 'y' datasets from the file
         x = np.array(f['group/x'], dtype=np.float32)
         y = np.array(f['group/y'], dtype=np.int8)

    return x


def display_voxel(voxels, color):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.indices(voxels.shape), c=color)
    plt.grid(b=None)
    plt.axis('off')
    plt.show()


# def display_features(features):
#     color_code = {0: "red", 1: "blue", 2: "green", 3: "orange",
#                   4: "grey", 5: "yellow", 6: "pink", 7: "purple"}
#     unique, counts = np.unique(features, return_counts=True)
#     colors = np.empty(features.shape, dtype=object)

#     for i in range(len(unique)):
#         colors[np.where(features == unique[i], True, False)] = color_code[i]

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.voxels(features, facecolors=colors)
#     plt.grid(b=None)
#     plt.axis('off')
#     plt.show()

def display_features(features):
    color_code = {0: "red", 1: "blue", 2: "green", 3: "orange",
                  4: "grey", 5: "yellow", 6: "pink", 7: "purple"}
    unique, counts = np.unique(features, return_counts=True)
    colors = np.empty(features.shape, dtype=object)

    for i in range(len(unique)):
        colors[np.where(features == unique[i], True, False)] = color_code[i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*np.indices(features.shape), c=colors)
    plt.grid(b=None)
    plt.axis('off')
    plt.show()




def display_machining_feature(index, count):
    feature_code = {0: "Chamfer", 1: "Through Hole", 2: "Triangular Passage", 3: "Rectangular Passage", 4: "6-Sided Passage",
                    5: "Triangular Through Slot", 6: "Rectangular Through Slot", 7: "Circular Through Slot",
                    8: "Rectangular Through Step", 9: "2-Sided Passage", 10: "Slanted Through Step",
                    11: "O-Ring", 12: "Blind Hole", 13: "Triangular Pocket",
                    14: "Rectangular Pocket", 15: "6-Sided Pocket", 16: "Circular End Pocket",
                    17: "Rectangular Blind Slot", 18: "Vertical Circular End Blind Slot", 19: "Horizontal Circular End Blind Slot",
                    20: "Triangular Blind Step", 21: "Circular Blind Step", 22: "Rectangular Blind Step", 23: "Round"}

    print(f"[{count}] -> Index: {index}, Machining Feature: {feature_code[index[0]]}")


if __name__ == '__main__':
    # User Parameters
    num_classes = 24
    voxel_resolution = 64
    binvox_path = "data/438.binvox"
    checkpoint_path = "checkpoint/featurenet_date_2020-12-12.ckpt"

    with open(binvox_path, 'rb') as f:
        model = binvox_rw.read_as_3d_array(f)
    voxel = model.data
    #display_voxel(voxel, "grey") # Display voxel model

    features = decomp_and_segment(voxel)

    model = FeatureNet(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.load_weights(checkpoint_path)

    for i, feature in enumerate(features):
        input = zero_centering_norm(feature)
        x = tf.Variable(input, dtype=tf.float32)
        x = tf.reshape(x, [1, 1, voxel_resolution,
        
                    voxel_resolution, voxel_resolution])
        test_logits = model(x, training= False)
        y_pred = np.argmax(test_logits.numpy(), axis=1)
        display_machining_feature(y_pred, i)
        # display_voxel(feature, "grey")
