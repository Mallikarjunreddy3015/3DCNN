import csv
import math
from stl import mesh


def link_facet_with_label(stl_mesh, facet_label):
    for i, vector in enumerate(stl_mesh.vectors):
        facet_label[i] = vector

    return facet_label


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


def get_facet_labels(facet_face_path, face_feature_path):
    facet_face = {}
    face_label = {}
    facet_label = {}

    with open(facet_face_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')

        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            facet_face[row[1]] = row[0]

    with open(face_feature_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=' ')

        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            face_label[row[0]] = row[6]

    for key, value in facet_face.items():
        facet_label[key] = face_label[value]

    return facet_label


if __name__ == '__main__':
    main_dir = "data/practice/"
    read_stl()