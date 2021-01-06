# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 06:37:35 2013

@author: Sukhbinder Singh

Source: http://sukhbinder.wordpress.com/2013/11/28/binary-stl-file-reader-in-python-powered-by-numpy/

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import h5py

from stl import mesh
from mpl_toolkits import mplot3d


def lines_to_voxels(line_list, pixels):
    for x in range(len(pixels)):
        is_black = False
        lines = list(find_relevant_lines(line_list, x))
        target_ys = list(map(lambda line: int(generate_y(line, x)), lines))
        for y in range(len(pixels[x])):
            if is_black:
                pixels[x][y] = True
            if y in target_ys:
                for line in lines:
                    if on_line(line, x, y):
                        is_black = not is_black
                        pixels[x][y] = True


def find_relevant_lines(line_list, x, ind=0):
    for line in line_list:
        same = False
        above = False
        below = False
        for pt in line:
            if pt[ind] > x:
                above = True
            elif pt[ind] == x:
                same = True
            else:
                below = True
        if above and below:
            yield line
        elif same and above:
            yield line


def generate_y(line, x):
    if line[1][0] == line[0][0]:
        return -1
    ratio = (x - line[0][0]) / (line[1][0] - line[0][0])
    y_dist = line[1][1] - line[0][1]
    new_y = line[0][1] + ratio * y_dist
    return new_y


def on_line(line, x, y):
    new_y = generate_y(line, x)
    if int(new_y) != y:
        return False
    if int(line[0][0]) != x and int(line[1][0]) != x and (max(line[0][0], line[1][0]) < x or min(line[0][0], line[1][0]) > x):
        return False
    if int(line[0][1]) != y and int(line[1][1]) != y and (max(line[0][1], line[1][1]) < y or min(line[0][1], line[1][1]) > y):
        return False
    return True


def to_intersecting_lines(stl_mesh, height):
    relevant_triangles = list(filter(lambda tri: is_above_and_below(tri, height), stl_mesh))
    not_same_triangles = filter(lambda tri: not is_intersecting_triangle(tri, height), relevant_triangles)
    lines = list(map(lambda tri: triangle_to_intersecting_lines(tri, height), not_same_triangles))
    return lines


def draw_line_on_pixels(p1, p2, pixels):
    line_steps = math.ceil(manhattan_distance(p1, p2))
    if line_steps == 0:
        pixels[int(p1[0]), int(p2[1])] = True
        return
    for j in range(line_steps + 1):
        point = linear_interpolation(p1, p2, j / line_steps)
        pixels[int(point[0]), int(point[1])] = True


def linear_interpolation(p1, p2, distance):
    '''
    :param p1: Point 1
    :param p2: Point 2
    :param distance: Between 0 and 1, Lower numbers return points closer to p1.
    :return: A point on the line between p1 and p2
    '''
    slope_x = (p1[0] - p2[0])
    slope_y = (p1[1] - p2[1])
    slope_z = p1[2] - p2[2]
    return (
        p1[0] - distance * slope_x,
        p1[1] - distance * slope_y,
        p1[2] - distance * slope_z
    )


def is_above_and_below(point_list, height):
    '''
    :param point_list: Can be line or triangle
    :param height:
    :return: true if any line from the triangle crosses or is on the height line,
    '''
    above = list(filter(lambda pt: pt[2] > height, point_list))
    below = list(filter(lambda pt: pt[2] < height, point_list))
    same = list(filter(lambda pt: pt[2] == height, point_list))
    if len(same) == 3 or len(same) == 2:
        return True
    elif above and below:
        return True
    else:
        return False


def is_intersecting_triangle(triangle, height):
    assert (len(triangle) == 3)
    same = list(filter(lambda pt: pt[2] == height, triangle))
    return len(same) == 3


def triangle_to_intersecting_lines(triangle, height):
    assert (len(triangle) == 3)
    above = list(filter(lambda pt: pt[2] > height, triangle))
    below = list(filter(lambda pt: pt[2] < height, triangle))
    same = list(filter(lambda pt: pt[2] == height, triangle))
    assert len(same) != 3
    if len(same) == 2:
        return same[0], same[1]
    elif len(same) == 1:
        side1 = where_line_crosses_z(above[0], below[0], height)
        return side1, same[0]
    else:
        lines = []
        for a in above:
            for b in below:
                lines.append((b, a))
        side1 = where_line_crosses_z(lines[0][0], lines[0][1], height)
        side2 = where_line_crosses_z(lines[1][0], lines[1][1], height)
        return side1, side2


def where_line_crosses_z(p1, p2, z):
    if p1[2] > p2[2]:
        t = p1
        p1 = p2
        p2 = t
    # now p1 is below p2 in z
    if p2[2] == p1[2]:
        distance = 0
    else:
        distance = (z - p1[2]) / (p2[2] - p1[2])
    return linear_interpolation(p1, p2, distance)


def calculate_scale_and_shift(stl_mesh, voxel_resolution):
    voxel_resolution -= 2
    all_points = [item for sublist in stl_mesh for item in sublist]
    mins = [0, 0, 0]
    maxs = [0, 0, 0]
    for i in range(3):
        mins[i] = min(all_points, key=lambda tri: tri[i])[i]
        maxs[i] = max(all_points, key=lambda tri: tri[i])[i]
    shift = [-m_value for m_value in mins]
    xyscale = (voxel_resolution - 1) / (max(maxs[0] - mins[0], maxs[1] - mins[1]))
    yzscale = (voxel_resolution - 1) / (max(maxs[1] - mins[1], maxs[2] - mins[2]))
    xzscale = (voxel_resolution - 1) / (max(maxs[0] - mins[0], maxs[2] - mins[2]))
    scale = [xyscale, yzscale, xzscale]
    x_scaled = math.ceil((maxs[0] - mins[0]) * min(scale))
    y_scaled = math.ceil((maxs[1] - mins[1]) * min(scale))
    z_scaled = math.ceil((maxs[2] - mins[2]) * min(scale))
                         
    bounding_box = [x_scaled, y_scaled, z_scaled]

    return min(scale), shift, bounding_box


def scale_and_shift_mesh(stl_mesh, scale, shift):
    for tri in stl_mesh:
        new_tri = []
        for pt in tri:
            new_pt = [0, 0, 0]
            for i in range(3):
                new_pt[i] = (pt[i] + shift[i]) * scale
            new_tri.append(tuple(new_pt))
        if len(remove_dups_from_point_list(new_tri)) == 3:
            yield new_tri
        else:
            pass

            
def manhattan_distance(p1, p2, d=2):
    assert (len(p1) == len(p2))
    all_distances = 0
    for i in range(d):
        all_distances += abs(p1[i] - p2[i])
    return all_distances


def print_big_array(big, yes='1', no='0'):
    print()
    for line in big:
        for char in line:
            if char:
                print(yes, end=" ")
            else:
                print(no, end=" ")
        print()


def remove_dups_from_point_list(pt_list):
    new_list = pt_list[:]
    return tuple(set(new_list))


def array_to_white_greyscale_pixel(array, pixels):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j]:
                pixels[i, j] = 255


def pad_voxel_array(voxels, voxel_resolution):
    voxel_resolution -= 2
    shape = voxels.shape
    padding = np.zeros((3, 2))
    
    for i in range(len(shape)):
        pad = voxel_resolution - shape[i]
        padding[i][0] = (pad // 2) + 1
        if pad % 2 == 0: 
            padding[i][1] = (pad // 2) + 1
        else: 
            padding[i][1] = (pad // 2) + 2
    
    padding_t = tuple(map(tuple, np.int_(padding)))
    pad_voxel = np.pad(voxels, padding_t, 'constant', constant_values=False)
    new_shape = pad_voxel.shape
    
    return pad_voxel, (new_shape[1], new_shape[2], new_shape[0])


def convert_to_voxel(stl_mesh, voxel_resolution=32):
    s_mesh = list(stl_vertices_generator(stl_mesh))
    (scale, shift, bounding_box) = calculate_scale_and_shift(s_mesh, voxel_resolution)
    s_mesh = list(scale_and_shift_mesh(s_mesh, scale, shift))

    # Note: vol should be addressed with vol[z][x][y]
    voxels = np.zeros((bounding_box[2], bounding_box[0], bounding_box[1]), dtype=bool)
    for height in range(bounding_box[2]):
        lines = to_intersecting_lines(s_mesh, height)
        prepixel = np.zeros((bounding_box[0], bounding_box[1]), dtype=bool)
        lines_to_voxels(lines, prepixel)
        voxels[height] = prepixel

    voxels, bounding_box = pad_voxel_array(voxels, voxel_resolution)

    return voxels


def zero_centering_norm(voxels):
    norm = (voxels - 0.5) * 2
    return norm


def write_h5_file(sample_num, split_name, voxels, y):
    file_name = split_name + '.h5'
    hf = h5py.File(file_name, 'a')
    group = hf.create_group(str(sample_num))
    group.create_dataset("x", data=voxels, compression="gzip", compression_opts=9)
    group.create_dataset("y", data=y)

    hf.close()


def read_stl(filepath, rotate=True):
    stl_mesh = mesh.Mesh.from_file(filepath)

    if rotate:
        stl_meshes = [mesh.Mesh(stl_mesh.data.copy()) for _ in range(6)]
        stl_meshes[1].rotate([-0.5, 0.0, 0.0], math.radians(90))
        stl_meshes[2].rotate([0.5, 0.0, 0.0], math.radians(90))
        stl_meshes[3].rotate([0.5, 0.0, 0.0], math.radians(180))
        stl_meshes[4].rotate([0.0, -0.5, 0.0], math.radians(90))
        stl_meshes[5].rotate([0.0, 0.5, 0.0], math.radians(90))
    else:
        stl_meshes = [stl_mesh]

    return stl_meshes


def stl_vertices_generator(stl_mesh):
    for i, j, k in zip(stl_mesh.v0, stl_mesh.v1, stl_mesh.v2):
        yield tuple(i), tuple(j), tuple(k)


def display_stl(filepath):
    # Create a new plot
    figure = plt.figure()
    axes = mplot3d.Axes3D(figure)

    # Load the STL files and add the vectors to the plot
    stl = mesh.Mesh.from_file(filepath)
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(stl.vectors))

    # Auto scale to the mesh size
    scale = stl.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    plt.show()


def display_voxel(voxels):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(voxels, edgecolor='k')
    plt.show()


if __name__ == '__main__':
    main_dir = "data/practice/"
    resolution = 64
    h5_file_name = f"multi_feature_voxels_{resolution}"

    normalise = False
    rotation = False


    sample_num = 0
    sub_dir_list = [f.path for f in os.scandir(main_dir) if f.is_dir()]

    for sub_dir in sub_dir_list:
        print(sub_dir[len(main_dir):])

        for file in os.listdir(sub_dir):
            try:
                file_path = sub_dir + "/" + file
                meshes = read_stl(file_path, rotate=rotation)
                y = int(sub_dir[len(main_dir):].split("_")[0])

                for m in meshes:
                    voxel = convert_to_voxel(m, voxel_resolution=resolution)

                    if normalise:
                        voxel = zero_centering_norm(voxel)

                    write_h5_file(sample_num, h5_file_name, voxel, y)
                    sample_num += 1

            except:
                print(file)
                continue

