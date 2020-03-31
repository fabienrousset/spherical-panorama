# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:44:42 2020

@author: admin
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from projection import draw_triangle
from drawing import draw_line


equi_src=mpimg.imread('Equirectangular_map.jpg')
plt.imshow(equi_src)
plt.show()




def distance(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))

def face_center(f):
    return np.mean(f, axis=0)

def apply_matrix(matrix, point):
    return np.matmul(matrix,point.reshape((point.shape[0],1))).transpose()[0]

def rotate(point, center, angle):
    translated = point - center
    rotation_matrix = np.array([[+np.cos(angle), -np.sin(angle)],
                                [+np.sin(angle), +np.cos(angle)]])
    rotated = apply_matrix(rotation_matrix, translated)
    return rotated + center


def symmetric(point, pointA, pointB):
    center = (pointA + pointB) / 2
    return point + 2*(center - point)

def dodecahedron(rotate=True):

    phi = (1+np.sqrt(5))/2

    # Define the dodecahedron vertices

    v00 = np.array([0, 0, 0])

    v01 = np.array([-1, -1, -1])
    v02 = np.array([-1, -1, +1])
    v03 = np.array([-1, +1, -1])
    v04 = np.array([-1, +1, +1])
    v05 = np.array([+1, -1, -1])
    v06 = np.array([+1, -1, +1])
    v07 = np.array([+1, +1, -1])
    v08 = np.array([+1, +1, +1])

    v09 = np.array([0, -phi, -1/phi])
    v10 = np.array([0, -phi, +1/phi])
    v11 = np.array([0, +phi, -1/phi])
    v12 = np.array([0, +phi, +1/phi])

    v13 = np.array([-1/phi, 0, -phi])
    v14 = np.array([-1/phi, 0, +phi])
    v15 = np.array([+1/phi, 0, -phi])
    v16 = np.array([+1/phi, 0, +phi])

    v17 = np.array([-phi, -1/phi, 0])
    v18 = np.array([-phi, +1/phi, 0])
    v19 = np.array([+phi, -1/phi, 0])
    v20 = np.array([+phi, +1/phi, 0])


    if rotate:

        f1 = np.array([v16, v14, v02, v10, v06])

        # Should be equal to
        # np.array([0, (-phi-2)/5, (3*phi+1)/5])
        f1_center = face_center(f1)

        # Should be equal to
        # np.array([0, (-phi-2)/5*np.sqrt(15/(4*phi+3)), (3*phi+1)/5*np.sqrt(15/(4*phi+3))])
        f1_pyramid = f1_center/distance(f1_center, v00)*np.sqrt(3)

        slope = (v16+v14)/2 - f1_pyramid
        angle = -np.arctan(slope[2]/slope[1])

        # Rotation around x axis
        rotation_matrix = np.array([[1,              0,              0],
                                    [0, +np.cos(angle), -np.sin(angle)],
                                    [0, +np.sin(angle), +np.cos(angle)]])

        # Pack, rotate and unpack
        vertices_3D = np.array([v01, v02, v03, v04, v05, v06, v07, v08, v09, v10,
                                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20])
        for i in range(20):
            vertices_3D[i] = apply_matrix(rotation_matrix, vertices_3D[i])
        v01, v02, v03, v04, v05, v06, v07, v08, v09, v10,\
        v11, v12, v13, v14, v15, v16, v17, v18, v19, v20 = vertices_3D


    # Define the faces
    f1 = np.array([v10, v06, v16, v14, v02])
    f2 = np.array([v05, v19, v06, v10, v09])
    f3 = np.array([v20, v08, v16, v06, v19])
    f4 = np.array([v12, v04, v14, v16, v08])
    f5 = np.array([v18, v17, v02, v14, v04])
    f6 = np.array([v01, v09, v10, v02, v17])

    return np.array([f1, f2, f3, f4, f5, f6])


def draw_pyramid(face, image, draw_center, draw_angle, draw_size, spherical=True):

    v00 = np.array([0, 0, 0])

    if spherical:
        # Define the center of the face
        center = face_center(face)
        # Define the tip of the pyramid
        tip = center/distance(center, v00)*distance(face[0], v00)
    else:
        # Resize the face to have edge=1
        face = face / distance(face[0], face[1])
        # Set the center
        center = face_center(face)
        # Define a radius
        radius = center/distance(center, v00)
        # Define the critical height of the pyramid
        h0 = np.sqrt(65+22*np.sqrt(5))/(19*np.sqrt(5))
        # Set the tip
        tip = center + h0*radius

    # Facet is a triangle
    facet_b = distance(face[0], face[1])
    facet_h = distance(tip, (face[0]+face[1])/2)

    display_info = False
    if display_info:
        h = distance(tip, center)
        h_norm = h / distance(face[0], v00)
        print(h)
        print(facet_b)
        print(facet_h)
        print(h_norm)
        print(facet_b/facet_h)
        print(facet_h/facet_b)

    facet_angle = 2 * np.arctan2(facet_b/2, facet_h)


    # Define the center
    center_2D = draw_center

    # Define the certices
    vertices_2D = np.zeros((6,2))

    # Set the first vertex
    vertices_2D[0] = draw_center + draw_size * np.array([0, 1])
    # Set all the vertices
    for i in range(1, 6):
        vertices_2D[i] = rotate(vertices_2D[i-1], center_2D, facet_angle)

    # Rotate to set base horizontal
    base_angle = np.arctan2(vertices_2D[3][1]-vertices_2D[2][1], vertices_2D[3][0]-vertices_2D[2][0])
    for i in range(0, 6):
        vertices_2D[i] = rotate(vertices_2D[i], draw_center, -base_angle)

    # Rotate to the specified angle
    for i in range(0, 6):
        vertices_2D[i] = rotate(vertices_2D[i], draw_center, draw_angle)



    triangles_2D = np.zeros((5, 3, 2))
    facets_3D = np.zeros((5, 3, 3))
    for i in range(5):
        triangles_2D[i] = np.array([center_2D, vertices_2D[i], vertices_2D[i+1]])
        facets_3D[i] = np.array([tip, face[i], face[(i+1)%5]])


    for i in range(5):
        print("Drawing triangle {}...".format(i+1))
        draw_triangle(facets_3D[i], triangles_2D[i], equi_src, image)
#        draw_line(image, triangles_2D[i][0], triangles_2D[i][1], [0,1,0])
#        draw_line(image, triangles_2D[i][0], triangles_2D[i][2], [0,1,0])
#        draw_line(image, triangles_2D[i][1], triangles_2D[i][2], [0,1,1])


    # Save the faces
    return triangles_2D


def draw_net(faces, image, draw_center, draw_size):

    # Draw central pyramid
    print("Drawing central pyramid...")
    triangles = draw_pyramid(faces[0], image, draw_center, 0, draw_size)
    
    # Draw side pyramids
    for i in range(5):
        print("Drawing pyramid on side {}...".format(i+1))
        new_center = symmetric(draw_center, triangles[i][1], triangles[i][2])
        angle = np.arctan2(triangles[i][1][1]-triangles[i][2][1], triangles[i][1][0]-triangles[i][2][0])
        draw_pyramid(faces[i+1], image, new_center, angle, draw_size)



def conversion(x, y, z):
    
    coordinates = np.array([x, y, z])

    # Radius to center
    radius = np.sqrt(coordinates[0]*coordinates[0]+coordinates[1]*coordinates[1]+coordinates[2]*coordinates[2])
    
    # Normalization
    coordinates = coordinates/radius
    
    # Conversion do spherical
    lmbda = np.arctan2(coordinates[1], coordinates[0])
    phi = np.arcsin(coordinates[2])
    

    return (lmbda, phi)

test = False
if test:
    test = np.ones((300, 300, 3))
    tri_3D = np.array([[+1, -1, -1], [+1, +2, -1], [+1, -1, +1]])
    tri_2D = np.array([[0, 0], [300, 0], [0, 200]])
    draw_triangle(tri_3D, tri_2D, equi_src, test)
    plt.imsave("test.png", test)
else:
    faces = dodecahedron()
    # Create the output images
    image_size = 1000
    position = np.array([image_size//2, image_size//2])
    draw_size = image_size/5.5
    image = np.ones((image_size, image_size, 3))
    #draw_pyramid(face, image, position, 0, image_size/2/1.2)
    #plt.imsave("pyramid.png", image)
    draw_net(faces, image, position, draw_size)
    plt.imsave("net_test.png", image)
    
