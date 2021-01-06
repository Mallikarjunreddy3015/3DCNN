import os
import sys


def parse_line(line_string):
    search_types = ["ADVANCED_FACE", "FACE_BOUND", "EDGE_LOOP", "MANIFOLD_SOLID_BREP", "CLOSED_SHELL"]

    split_line = line_string.split()

    try:
        idx = split_line[0][1:]

        if search_types[0] in split_line[2]:
            parsed_attribute = split_line[2][len(search_types[0]):]
            face_bound_idx = parse_advanced_face(parsed_attribute)
            print(f"Face: {idx} bounded by Face Bound {face_bound_idx}")

        elif search_types[1] in split_line[2]:
            parsed_attribute = split_line[2][len(search_types[1]):]
            edge_loop_idx = parse_face_bound(parsed_attribute)
            print(f"Face bound {idx} bounded by Edge Loop {edge_loop_idx}")

        elif search_types[2] in split_line[2]:
            parsed_attribute = split_line[2][len(search_types[1]):]
            oriented_edges = parse_edge_loop(parsed_attribute)
            print(f"Edge Loop {idx} contains edges {oriented_edges}")

        elif search_types[3] in split_line[2]:
            parsed_attribute = split_line[2][len(search_types[3]):]
            closed_shell_idx = parse_manifold_solid_brep(parsed_attribute)
            print(f"\nSolid {idx} contains closed shell {closed_shell_idx}")

        elif search_types[4] in split_line[2]:
            parsed_attribute = split_line[2][len(search_types[4]):]
            adv_face_idxs = parse_closed_shell(parsed_attribute)
            print(f"Closed Shell {idx} contains advanced faces {adv_face_idxs}")

    except:
        return


def parse_manifold_solid_brep(attribute):
    attribute_split = attribute.split(",")
    closed_shell_index = attribute_split[1][1:-2]

    return closed_shell_index


def parse_closed_shell(attribute):
    attribute = attribute.replace(",", "")
    attribute = attribute.replace("(", "")
    attribute = attribute.replace(")", "")
    attribute = attribute.replace(";", "")
    attribute = attribute.replace("'", "")
    attribute = attribute.replace(" ", "")

    adv_face_idxs = attribute.split("#")[1:]

    return adv_face_idxs


def parse_advanced_face(attribute):
    attribute_split = attribute.split(",")
    face_bound_index = attribute_split[1][2:-1]

    return face_bound_index


def parse_face_bound(attribute):
    attribute_split = attribute.split(",")
    face_bound_index = attribute_split[1][1:]

    return face_bound_index


def parse_edge_loop(attribute):
    attribute = attribute.replace(",", "")
    attribute = attribute.replace("(", "")
    attribute = attribute.replace(")", "")
    attribute = attribute.replace(";", "")
    attribute = attribute.replace("'", "")
    attribute = attribute.replace(" ", "")

    oriented_edge_idxs = attribute.split("#")[1:]

    return oriented_edge_idxs


if __name__ == '__main__':
    is_data = False

    with open('0-6-11-24.step', 'r') as file:
        for i, line in enumerate(file):
            if line == "DATA;\n":
                is_data = True
                continue

            if is_data:
                parse_line(line)
