# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:10:43 2020

@author: admin
"""

import numpy as np

def set_pixel_value(img, x, y, value):
    if x<0 or x>=img.shape[1] or y<0 or y>=img.shape[0]:
        return
    else:
        img[img.shape[0]-1-y][x] = value

def get_pixel_value(img, x, y):
    if x<0 or x>=img.shape[1] or y<0 or y>=img.shape[0]:
        return [1, 0, 1]
    else:
        return img[img.shape[0]-1-y][x]

def draw_line(img, pointA, pointB, value, n_dots=1000):
    for k in np.linspace(0,1,2+n_dots):
        point = (k*pointA + (1-k)*pointB).astype(int)
        set_pixel_value(img, point[0], point[1], value)

