import tensorflow as tf
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from network import FeatureNet
import numpy as np
import h5py
import matplotlib.pyplot as plt
import utils.binvox_rw as binvox_rw
from skimage import measure


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
    blobs = ~sample
    final_labels = np.zeros(blobs.shape)
    all_labels = measure.label(blobs)

    for i in range(1, np.max(all_labels) + 1):
        mk = (all_labels == i)
        distance = ndi.distance_transform_edt(mk)

        labels = watershed(-distance)

        max_val = np.max(final_labels) + 1
        idx = np.where(mk)

        final_labels[idx] += (labels[idx] + max_val)

    results = get_seg_samples(final_labels)

    return results


def decomp_and_segment_2(vox):
    machining_features = []
    components = []
    color_code = {0: "red", 1: "blue", 2: "green", 3: "orange", 4: "grey", 5: "yellow"}
    inverted_voxel = np.logical_not(vox).astype(np.int)

    # Get connected regions
    markers = ndi.label(inverted_voxel)

    for i in range(markers[1] + 1):
        if i == 0:
            continue
        else:
            feature = np.where(markers[0] == i, 1, 0)
            components.append(feature)

    for component in components:
        component = np.logical_not(component).astype(np.int)
        #bbox = get_bounding_box(vox)
        #test_voxel = component[bbox[0]:bbox[1], bbox[2]:bbox[3], bbox[4]:bbox[5]]
        distance = ndi.distance_transform_edt(component)

        max_d = np.max(distance)
        colors = np.empty(component.shape, dtype=object)
        colors[np.where(distance <= max_d, True, False)] = color_code[0]
        colors[np.where(distance < max_d * 0.8, True, False)] = color_code[1]
        colors[np.where(distance < max_d * 0.6, True, False)] = color_code[2]
        colors[np.where(distance < max_d * 0.4, True, False)] = color_code[3]
        colors[np.where(distance < max_d * 0.2, True, False)] = color_code[4]
        colors[np.where(distance < max_d * 0.05, True, False)] = color_code[5]

        local_maxi = peak_local_max(distance, min_distance=1, indices=False, labels=component)
        print(np.max(local_maxi))

        if np.max(local_maxi) == False:
            machining_features.append(component)
            continue

        component = np.logical_not(component).astype(np.int)
        distance = ndi.distance_transform_edt(component)
        local_maxi = peak_local_max(distance, min_distance=7, indices=False, labels=component)

        markers = ndi.label(local_maxi)[0]
        labels = watershed(-distance, markers, mask=component, compactness=10)
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Detected Labels: {dict(zip(unique, counts))}")

        for i in unique:
            if i == 0:
                continue
            segmented_feature = np.where(labels == i, 1, 0)
            #display_voxel(segmented_feature, "green")
            segmented_feature = np.logical_not(segmented_feature).astype(np.int)
            machining_features.append(segmented_feature)

    return machining_features


def get_bounding_box(voxel):
    a = np.where(voxel != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
    return bbox


def test_step(x):
    test_logits = model(x, training=False)
    y_pred = np.argmax(test_logits.numpy(), axis=1)
    return y_pred


def zero_centering_norm(voxels):
    norm = (voxels - 0.5) * 2
    return norm


def load_voxel(file_path):
    hf = h5py.File(file_path, 'r')
    x = None

    for key in list(hf.keys()):
        group = hf.get(key)
        x = np.array(group.get("x"), dtype=np.float32)
        y = np.array(group.get("y"), dtype=np.int8)

    return x


def display_voxel(voxels, color):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, facecolors=color)
    #plt.grid(b=None)
    #plt.axis('off')
    plt.show()


def display_machining_feature(index, count):
    feature_code = {0: "O-Ring", 1: "Through Hole", 2: "Blind Hole", 3: "Triangular Passage", 4: "Rectangular Passage",
                    5: "Circular Through Slot", 6: "Triangular Through Slot", 7: "Rectangular Through Slot",
                    8: "Rectangular Blind Slot", 9: "Triangular Pocket", 10: "Rectangular Pocket",
                    11: "Circular End Pocket", 12: "Triangular Blind Step", 13: "Circular Blind Step",
                    14: "Rectangular Blind Step", 15: "Rectangular Through Step", 16: "2 Sides Through Step",
                    17: "Slanted Through Step", 18: "Chamfer", 19: "Round", 20: "Vertical Circular End Blind Slot",
                    21: "Horizontal Circular End Blind Slot", 22: "6 Sides Passage", 23: "6 Sides Pocket"}

    print(f"[{count}] -> Index: {index}, Machining Feature: {feature_code[index[0]]}")


if __name__ == '__main__':
    import os
    import glob
    resolution = 64

    with open("data/practice/0-0-5-14-15.binvox", "rb") as f:
        model = binvox_rw.read_as_3d_array(f)
    voxel = model.data

    #region_labels, unique_labels = decomp_and_segment(voxel)
    #features = decomp_and_segment_2(voxel)
    features = decomp_and_segment(voxel)

    num_classes = 24
    num_epochs = 50
    learning_rate = 0.001
    decay_rate = learning_rate / num_epochs
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,
                                                                 decay_steps=100000, decay_rate=decay_rate)

    model = FeatureNet(num_classes=num_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    model.load_weights("checkpoint/featurenet_date_2020-11-06.ckpt")

    for i, feature in enumerate(features):
        #feature = pad_voxel_array(feature, resolution)
        input = zero_centering_norm(feature)
        x = tf.Variable(input, dtype=tf.float32)
        x = tf.reshape(x, [1, 1, resolution, resolution, resolution])
        y_pred = test_step(x)
        display_machining_feature(y_pred, i)
        #display_voxel(feature, "grey")



    """
     for i in unique_labels:
        if i == 0:
            continue
        feature = np.where(region_labels == i, 1, 0)
        feature = np.ones(np.shape(region_labels)) - feature
        feature = pad_voxel_array(feature, resolution)

        equal_array = np.equal(voxel, feature)
        invert = np.logical_not(equal_array).astype(np.int)
        print(np.count_nonzero(invert) / 64 / 64 / 64 * 100)
    
    for i in unique_labels:
        if i == 0:
            continue
        feature = np.where(region_labels == i, 1, 0)
        feature = np.ones(np.shape(region_labels)) - feature
        feature = feature.astype(dtype=np.bool)
        input = zero_centering_norm(feature)

        x = tf.Variable(input, dtype=tf.float32)
        x = tf.reshape(x, [1, 1, resolution, resolution, resolution])
        y_pred = test_step(x)
        display_machining_feature(y_pred, i)
        display_voxel(feature, "white")   
    """




    """
    dirpath = "data/voxel/"
    new_dir_path = "data/voxel_final/"

    for i in os.listdir(dirpath):
        sub_dir = dirpath + i + "/"
        list_of_files = glob.glob(sub_dir + "*.binvox")

        for file in list_of_files:
            print(file)
            print(new_dir_path + file[len(sub_dir):])
            #os.rename(sub_dir + file, new_dir_path + file)    
    """
"""
    test_condition = False
    feature_num = 9
    feature = str(feature_num) + "_" + "triangular_pocket/"
    voxel_dirpath = "data/voxel_finish/" + feature
    stl_dirpath = "data/stl/" + feature
    #voxel = load_voxel("multi_feature_voxels_64.h5")
    #voxel = voxel.astype(np.int)
    count = 0

    print("Test: ", test_condition)
    for file in glob.glob(dirpath + "*.binvox"):
        filename = file[len(dirpath):]

        try:
            label = int(filename.split("_")[0])
            if label == feature_num:
                count += 1
                #print(filename)
                if not test_condition:
                    os.rename(file, voxel_dirpath + filename)
        except Exception as e:
            print("Error: ", filename)
            print(e)
            break
    print(count)

    count = 0
    for file in glob.glob(dirpath + "*.STL"):
        filename = file[len(dirpath):]

        try:
            label = int(filename.split("_")[0])
            if label == feature_num:
                count += 1
                #print(filename)
                if not test_condition:
                    os.rename(file, stl_dirpath + filename)
        except Exception as e:
            print("Error: ", filename)
            print(e)
            break
    print(count)   
"""

