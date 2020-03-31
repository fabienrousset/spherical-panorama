# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:08:53 2020

@author: admin
"""
import numpy as np
import matplotlib.pyplot as plt

from drawing import set_pixel_value, get_pixel_value

def orthographic_xy_to_lp(x, y, lmbda0, phi0, img):
    
    shape = img.shape
    
    if shape[0] != shape[1]:
        print("Orthographic image should be square")
        return (np.nan, np.nan)

    R = shape[0]/2
        
    rho = np.sqrt(x*x + y*y)
    if rho > R:
#        print("Point outside of projection")
        return (np.nan, np.nan)
    c = np.arcsin(rho/R)
    
    if rho==0:
        return (lmbda0, phi0)
    
    phi = np.arcsin(np.cos(c) * np.sin(phi0) + y*np.sin(c)*np.cos(phi0)/rho)
    lmbda = lmbda0 + np.arctan2(x*np.sin(c),(rho*np.cos(c)*np.cos(phi0)-y*np.sin(c)*np.sin(phi0)))
    
    return (lmbda, phi)

def gnomonic_xy_to_lp(x, y, lmbda0, phi0, img, hfov):
        
    x0 = -np.cos(phi0)*np.cos(lmbda0)
    y0 = -np.cos(phi0)*np.sin(lmbda0)
    z0 = +np.sin(phi0)
    
    print(x0, y0, z0)
    

def project_face_rect(face, out, equi_src):

    """Wrong orientation"""
    
    v1, v2, v3, v4 = face
    x_max, y_max, _ = out.shape

    for x in range(x_max):
        for y in range(y_max):

            # Reduced pixel coordinates
            x1, y1 = x/x_max, y/y_max            
            
            # 3D point interpolation
            coordinates = (1-y1)*(1-x1)*v1 + y1*(1-x1)*v2 + y1*x1*v3 + (1-y1)*x1*v4
            
            # Radius to center
            radius = np.sqrt(coordinates[0]*coordinates[0]+coordinates[1]*coordinates[1]+coordinates[2]*coordinates[2])

            # Normalization
            coordinates = coordinates/radius
            
            # Conversion do spherical
            lmbda = np.arctan2(coordinates[1], -coordinates[0])
            phi = np.arcsin(coordinates[2])
            
            # Conversion to equirectangular
            x2, y2 = equirectangular_lp_to_xy(lmbda, phi, 0, 0, equi_src)
            
            # Apply color
            if np.isnan(x2) or np.isnan(y2):
                set_pixel_value(out, x, y, [0, 0, 0])
            else:
                value = get_pixel_value(equi_src, int(x2), int(y2))/255
                set_pixel_value(out, x, y, value)

            
            
def equirectangular_lp_to_xy(lmbda, phi, lmbda0, phi0, img):
    
    shape = img.shape
    
    if shape[0] != shape[1]//2:
        print("Equirectangular image should be 1:2 aspect ratio")
        return (np.nan, np.nan)
    
    if np.isnan(lmbda) or np.isnan(phi):
        return (np.nan, np.nan)

    x = ((1/2 + lmbda/(2*np.pi)) * shape[1])%shape[1]
    y = ((1/2 + phi/np.pi) * shape[0])%shape[0]
    
    return (x, y)


def orthographic(lmbda, phi, equi_src):
    map_size = 400
    view = np.zeros((map_size, map_size, 3))
        
    for x in range(map_size):
        for y in range(map_size):
            
            lmbda, phi = orthographic_xy_to_lp(x-map_size//2, y-map_size//2, lmbda, phi, view)
            x1, y1 = equirectangular_lp_to_xy(lmbda, phi, 0, 0, equi_src)
                    
            if np.isnan(x1) or np.isnan(y1):
                set_pixel_value(view, x, y, [0, 0, 0])
            else:
                value = get_pixel_value(equi_src, int(x1), int(y1))/255
                set_pixel_value(view, x, y, value)
    
    plt.imshow(view)
    plt.show()
    

def cube_faces(equi_src):
    
    # Define the vertices of the cube
    v1 = np.array([-1, -1, -1])
    v2 = np.array([-1, -1, +1])
    v3 = np.array([-1, +1, -1])
    v4 = np.array([-1, +1, +1])
    v5 = np.array([+1, -1, -1])
    v6 = np.array([+1, -1, +1])
    v7 = np.array([+1, +1, -1])
    v8 = np.array([+1, +1, +1])
    
    # Define the faces of the cube
    face1 = np.array([v1, v2, v4, v3])
    face2 = np.array([v3, v4, v8, v7])
    
    # Create the output images
    face_size = 200
    out1 = np.zeros((face_size, face_size, 3))
    out2 = np.zeros((face_size, face_size, 3))

    # Project onto the faces
    project_face_rect(face1, out1, equi_src)
    project_face_rect(face2, out2, equi_src)

    # Save the faces
    plt.imsave("face_1.png", out1)
    plt.imsave("face_2.png", out2)


def barycentric_2D(triangle, point):
    T = np.stack((triangle[0]-triangle[2], triangle[1]-triangle[2]), axis = 1)
    barycentric = np.matmul(np.linalg.inv(T), (point-triangle[2]).reshape((2,1)))
    return barycentric.transpose()[0]


def draw_triangle(tri_3D, tri_2D, equi_src, out):

    if tri_3D.shape != (3, 3):
        print("Wrong 3D triangle shape")
    if tri_2D.shape != (3, 2):
        print("Wrong 2D triangle shape")
    
    # Refine the area to process
    x_min = int(np.min(tri_2D[:,0]))
    x_max = int(np.max(tri_2D[:,0]))
    y_min = int(np.min(tri_2D[:,1]))
    y_max = int(np.max(tri_2D[:,1]))
    
    
    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            
            # Barycentric coordinates in the output image
            barycentric = barycentric_2D(tri_2D, np.array([x,y]))
            
            if barycentric[0]>=0 and barycentric[0]<=1 and\
               barycentric[1]>=0 and barycentric[1]<=1 and\
               1-barycentric[0]-barycentric[1]>=0 and\
               1-barycentric[0]-barycentric[1]<=1:
                   
                
                # 3D point interpolation
                coordinates = barycentric[0]*tri_3D[0] +\
                              barycentric[1]*tri_3D[1] +\
                              (1-barycentric[0]-barycentric[1])*tri_3D[2]
                
                # Radius to center
                radius = np.sqrt(coordinates[0]*coordinates[0]+coordinates[1]*coordinates[1]+coordinates[2]*coordinates[2])
                
                # Normalization
                coordinates = coordinates/radius
                
                # Conversion do spherical
                lmbda = np.arctan2(coordinates[1], coordinates[0])
                phi = np.arcsin(coordinates[2])
                
                # Conversion to equirectangular
                x2, y2 = equirectangular_lp_to_xy(lmbda, phi, 0, 0, equi_src)
                
                # Apply color
                if np.isnan(x2) or np.isnan(y2):
                    set_pixel_value(out, x, y, [1, 0, 1])
                else:
                    value = get_pixel_value(equi_src, int(x2), int(y2))/255
                    # Debug value:
                    # value*np.array([barycentric[0], barycentric[1], 1-barycentric[0]-barycentric[1]])
                    # For use when debugging
                    set_pixel_value(out, x, y, value)

