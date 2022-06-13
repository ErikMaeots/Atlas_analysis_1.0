import pandas as pd
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
# import cucim.skimage.transform as cp
import cupyx.scipy.ndimage as cpx
import cucim.skimage.morphology as cumorph
import cucim.skimage.filters as cufilter
import cucim.skimage.exposure as cuexposure
import cucim.skimage.feature as cufeature
from cupyx.scipy.signal import convolve2d
# from scipy.interpolate import interp2d
# from skimage import exposure
from tensorflow import keras
# import tensorflow as tf
# import skimage.draw as draw
import logging
import matplotlib as cm
# import re
# import argparse
import multiprocessing as mp
# from time import sleep
import tensorflow as tf
import skimage.draw as draw
# import math
import os
import glob
# import cupy
import cv2
nfilters = 16
cm.rcParams.update({'font.size': 10})
# parser = argparse.ArgumentParser(description='Returns Ice thickness estimates based on modelf W:/home/shared/Machine_Learning_Datasets/All_datasets_with_atlas_alignment/atlas_model_test532')
# parser.add_argument("-input", help='Filename that contains list of tiles to analyse', default='_atlas.csv')
# parser.add_argument("-output", help='Name associated with output files', default='gNEXXX')
# parser.add_argument("-mode", help='training, inference, or test', default='inference')
# parser.add_argument("-threads", help='nr of threads to work, 1-3 recommended', default=3)
# args = parser.parse_args()
class Dataset:
    def __init__(self, grids, tiles, datasets, ice_thickness, doses):
        self.grids = grids
        self.tiles = tiles
        self.datasets = datasets
        self.ice_thickness = ice_thickness
        self.doses = doses
    def add_grid(self, grid):
        self.grids.append(grid)
    def add_tile(self, tile):
        self.tiles.append(tile)
    def add_dataset(self, dataset):
        if self.datasets is None:
            self.datasets = dataset
            # print(self.datasets.shape)
        else:
            self.datasets = tf.concat([self.datasets, dataset], 0 )
            # print(self.datasets.shape)
    def add_doses(self, doses):
        if self.doses is None:
            self.doses = doses
        else:
            self.doses = pd.concat([self.doses, doses])
            # print('added up', len(self.doses))
class Grid:
    def __init__(self, name, meta_file, hole_size, hole_spacing, blanks, square_angle, hole_angle):
        self.name = name
        self.meta_file = meta_file
        self.hole_size = hole_size
        self.hole_spacing = hole_spacing
        self.blanks = blanks
        self.square_angle = square_angle
        self.hole_angle = hole_angle

    def add_hole_size(self, size):
        self.hole_size.append(size)
    def get_hole_size(self):
        if len(self.hole_size) > 1:
            return np.mean(self.hole_size), np.std(self.hole_size)
        else:
            return 110, 10 ## defaults to result in target size 50
    def get_radius(self):
        return np.sqrt(np.mean(self.hole_size) / np.pi)
    def add_hole_spacing(self, spacing):
        self.hole_spacing.append(spacing)
    def get_hole_spacing(self):
        if len(self.hole_spacing) > 3:
            print(self.hole_spacing)
            std = np.std(self.hole_spacing)
            mean = np.mean(self.hole_spacing)
            if std < 0.5:
                std = 0.5
            elif std > 10:
                std = 10
            elif not 17 < mean < 55:
                mean = 36
            return mean, std
        else:
            return self.hole_spacing[0], 3 ## defaults to std=3
    def add_blanks(self, blank):
        self.blanks.append(blank)
    def get_blanks(self):
        return np.mean(np.asarray(self.blanks)), np.std(self.blanks)
    def add_square_angle(self, angle):
        self.square_angle.append(angle)
    def get_square_angle(self):
        return np.mean(self.square_angle)
    def add_hole_angle(self, angle):
        self.hole_angle.append(angle)
    def get_hole_angle(self):
        if len(self.hole_angle) > 3:
            std = np.std(self.hole_angle)
            if std < 0.5:
                std = 0.5
            if std < 22.5:
                std = 22.5
            return np.mean(self.hole_angle), std
        else:
            return 45, 45 ## defaults to 90 degrees spatial search
    def get_list_of_tiles(self):
        return self.meta_file['filename']
class Tile(Grid):
    def __init__(self, name, meta_file, hole_size, hole_spacing, blanks, square_angle, hole_angle, filename,
                 local_blanks, image, coordinates, binary_image, doses, ice_thickness):
        super().__init__(name, meta_file, hole_size, hole_spacing, blanks, square_angle, hole_angle)
        self.filename = filename
        self.local_blanks = local_blanks
        self.image = image
        self.coordinates = coordinates
        self.binary_image = binary_image
        self.doses = doses
        self.ice_thickness = ice_thickness
    def add_local_blanks(self, blank):
        self.local_blanks.append(blank)
    def get_offset(self, Grid):
        if len(self.local_blanks) > 0:
            return np.mean(self.local_blanks)
        elif len(Grid.blanks) > 0:
            return np.mean(Grid.blanks)
        else:
            return 230

    def get_zero(self):
        if len(self.local_blanks) > 1:
            return np.mean(self.local_blanks)
        elif len(self.blanks) > 0:
            return np.mean(self.blanks)
        else:
            return np.max(self.image)

def apply_regex_to_df(df, to_replace, string):
    return df.replace(to_replace=to_replace, value=string, regex=True)
def start_Grid_class(input, output):
    meta_file = pd.read_csv(input)
    meta_file = apply_regex_to_df(meta_file, '.xml', '.mrc')
    # meta_file = apply_regex_to_df(meta_file, '/camp', 'W:')
    # meta_file = apply_regex_to_df(meta_file, '/', r"\'")
    grid = output
    return Grid(grid, meta_file, [], [], [], [], [])
def start_Tile_class(Grid, tile):
    grid = Grid.name
    meta_file = Grid.meta_file
    hole_size = Grid.hole_size
    hole_spacing = Grid.hole_spacing
    blanks = Grid.blanks
    square_angle = Grid.square_angle
    hole_angle = Grid.hole_angle
    local_blanks = []
    image = load_image(tile)
    return Tile(grid, meta_file, hole_size, hole_spacing,
                blanks, square_angle, hole_angle, tile, local_blanks, image, [], None, None, None)

def load_image(filename, new_size=0, k=1):
  with mrcfile.open(filename) as f:
    f = cp.array(f.data)
    f = cp.rot90(f, k=k)
    f = cp.fliplr(f)
    f = f * (255 / f.max())
    return f.astype(cp.int32)

def cupy_calc_size_distance(image, thres, target_size, special=False):
    blob, list_sum, empties = find_holes(image, target_size, thres, special=special)
    blob, num_dots = cpx.label(blob)
    # plt.imshow(blob.get())
    # plt.show()
    list_index = cp.arange(1, num_dots + 1)
    try:
        dot_size = cp.asarray(list_sum)
        dot_size = cp.median(dot_size)
        list_cent = cp.asarray(cpx.measurements.center_of_mass(blob, labels=blob, index=list_index))

        list_dist = [cp.sort(cp.sqrt((dot[0] - list_cent[:, 0]) ** 2
                                   + (dot[1] - list_cent[:, 1]) ** 2))[1]
                   for dot in list_cent]
        dot_dist = cp.median(cp.array(list_dist))
        return dot_size, dot_dist, list_cent, blob, np.asarray(empties)
    except:
        print('EXCEPTION')
        return 0,0,[],0,[]

def find_holes(image, target_size, thres, special):
    array = cpx.median_filter(image, size=(2, 2))
    array = cuexposure.rescale_intensity(array, in_range=(100, 220))
    array = array * (255 / array.max())
    hist, edges = cp.histogram(array[cp.nonzero(array)], bins=256)
    th = cufilter.threshold_otsu(hist=hist)
    binary = cp.asarray(array > th, dtype=np.float32)
    binary = cpx.binary_erosion(binary, iterations=3, brute_force=True)
    binary = cpx.binary_dilation(binary, iterations=3, brute_force=True)
    # binary = cumorph.closing(binary, cumorph.disk(1))
    # binary = cumorph.opening(binary, cumorph.disk(1))
    lbls, nr_of_lbls = cpx.label(binary)
    # plt.imshow(binary.get())
    # plt.show()
    lbl = cp.arange(1, nr_of_lbls + 1)
    list_of_empty = []
    list_of_sizes = []
    if nr_of_lbls < 200:
        new_array = array * binary
        hist, edges = cp.histogram(new_array[cp.nonzero(new_array)], bins=256)
        th = cufilter.threshold_otsu(hist=hist)
        binary = cp.asarray(array > th, dtype=np.float32)
        binary = cumorph.closing(binary, cumorph.disk(1))
        binary = cumorph.opening(binary, cumorph.disk(1))
        # plt.imshow(binary.get())
        # plt.show()
        lbls, nr_of_lbls = cpx.label(binary)
        lbl = cp.arange(1, nr_of_lbls + 1)
        print('fallback binarization found', nr_of_lbls)
        for l in lbl:
          subarray = lbls[lbls == l]
          size = subarray.size
          if size < target_size:
              lbls[lbls == l] = 0
          if size > target_size * 8:
              org_slice = image[lbls == l]
              empty_mean = cp.mean(org_slice)
              list_of_empty.append(empty_mean.get())
              lbls[lbls == l] = 0
          else:
              lbls[lbls == l] = 1
              list_of_sizes.append(size)
        return lbls, list_of_sizes, list_of_empty
    if special:

        for l in lbl:
            subarray = lbls[lbls == l]
            size = subarray.size

            if size < target_size:
                lbls[lbls == l] = 0
            if size > target_size * 8:
                org_slice = image[lbls == l]
                empty_mean = cp.mean(org_slice)
                list_of_empty.append(empty_mean.get())

                lbls[lbls == l] = 1
            else:
                list_of_sizes.append(size)
                lbls[lbls == l] = 0
        return lbls, list_of_sizes, list_of_empty
    else:

        for l in lbl:
          subarray = lbls[lbls == l]
          size = subarray.size
          if size < target_size:
              lbls[lbls == l] = 0
          if size > target_size * 8:
              org_slice = image[lbls == l]
              empty_mean = cp.mean(org_slice)
              list_of_empty.append(empty_mean.get())
              lbls[lbls == l] = 0
          else:
              lbls[lbls == l] = 1
              list_of_sizes.append(size)
        return lbls, list_of_sizes, list_of_empty

def find_gridsquares(image, th, mode=None):
  ### preprocesses image to get binary mask of each square, then return the masked image and the corners of each square
  eroded = cp.asarray(image > th, dtype=cp.float32)
  eroded = cpx.binary_dilation(eroded, iterations=20, brute_force=True)
  if mode == 'hex':
      eroded = cpx.binary_erosion(eroded, iterations=25, brute_force=True)
  else:
      eroded = cpx.binary_erosion(eroded, iterations=15, brute_force=True)
  lbls, nr_of_lbls = cpx.label(eroded)
  lbl = cp.arange(1, nr_of_lbls + 1)
  for l in lbl:
    subarray = lbls[lbls == l]
    size = subarray.size
    if size < 10000:
      lbls[lbls == l] = 0
      # print('too small')
    else:
      lbls[lbls == l] = 1

  lbls = cp.asarray(lbls == 0, dtype=cp.float32)
  lbls, nr_of_lbls = cpx.label(lbls)
  lbl = cp.arange(1, nr_of_lbls + 1)
  for l in lbl:
    subarray = lbls[lbls == l]
    size = subarray.size
    if size < 50000:  ## Fills in dark areas within squares i.e ice
      lbls[lbls == l] = 0
      # print('too small')
    else:
      lbls[lbls == l] = 1
  lbls = cp.asarray(lbls == 0, dtype=cp.float32)
  lbls = lbls.astype(cp.uint8)
  contours, hier = cv2.findContours(lbls.get(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  boxes = []
  angles = []
  for c in contours:
    rect = cv2.minAreaRect(c)
    a,b,angle = rect
    angles.append(angle)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    boxes.append(box)
  boxes = np.concatenate(boxes, axis=1)
  lbls, nr_of_labels = cpx.label(lbls)
  square_outlines = repack_corners(boxes, nr_of_labels)
  return lbls, square_outlines, angles


def repack_corners(corners, nr_of_corners):
    all_shapes = []
    n = 0
    for it in range(nr_of_corners):
        try:
            x = corners[:, n]
            y = corners[:, n + 1]
            vertices = np.array([(x[0], y[0]), (x[1], y[1]), (x[2], y[2]), (x[3], y[3])])
            all_shapes.append(vertices)
            n += 2
        except:
            pass
    return cp.array(all_shapes)

def check_extent(x, low=0, high=4095):
    if x < low:
        x = cp.array(low)
    elif x > high:
        x = cp.array(high)
    return x
def crop_center(img,cropx,cropy, offx=0, offy=0):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    x1 = check_extent((startx-offx), 0, x)
    x2 = check_extent((startx+cropx-offx), 0, x)
    y1 = check_extent((starty-offy), 0, x)
    y2 = check_extent((starty+cropy-offy), 0, y)
    return img[y1:y2, x1:x2]

def align_image(tile, square):
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(tile.get())
    # ax[1].imshow(square.get())
    # plt.show()
    result = cufeature.match_template(tile, square)
    top_left_y, top_left_x = cp.unravel_index(cp.argmax(result), cp.shape(result))
    return top_left_x, top_left_y, cp.max(result)
def align_points(image ,points, vertices, xsize, ysize, xdot_dist, ydot_dist, final=False):
    if points is None:
        return cp.array(0)
    x1 = check_extent(vertices[:, 0].min() - xdot_dist)
    x2 = check_extent(x1 + xsize + (2 * xdot_dist))
    y1 = check_extent(vertices[:, 1].min() - ydot_dist)
    y2 = check_extent(y1 + ysize + (2 * ydot_dist))
    test_img = image[y1:y2, x1:x2]
    # test_img = test_img / test_img.max()
    # points = cp.asarray(points > 0.5, dtype=cp.float32)
    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image.get())
    # ax[1].imshow(points.get())
    # plt.show()
    top_left_x, top_left_y, alignment = align_image(test_img, points)
    if final:
        eroded = cp.asarray(points > 0.5, dtype=cp.float32)
        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(test_img.get())
        # ax[1].imshow(points.get())
        # ax[2].imshow(test_img.get())
        # ax[2].imshow(points.get(), alpha=0.2)
        # plt.show()
        lbls, nr_of_lbls = cpx.label(eroded)
        list_index = cp.arange(1, nr_of_lbls + 1)
        list_cent = cp.asarray(cpx.measurements.center_of_mass(lbls, labels=lbls, index=list_index))
        list_cent = pd.DataFrame(data=list_cent.get(), columns=['Y', 'X'])

        list_cent['X'] = list_cent['X'] + x1.get() + top_left_x.get()
        list_cent['Y'] = list_cent['Y'] + y1.get() + top_left_y.get()
        return list_cent
    else:
        return alignment
# def serious_pattern(radius, xspacing, yspacing, theta, xsize, ysize):
#     xspace = cp.arange(2, int(xsize.get()), int(xspacing.get()))
#     yspace = cp.arange(2, int(ysize.get()), int(yspacing.get()))
#     xs, ys = cp.meshgrid(yspace, xspace)
#     grid = cp.vstack([xs.ravel(), ys.ravel()])
#     fake_image = cp.zeros((int(ysize), int(xsize)))
#     def new_kernel(radius):
#         kernel = cumorph.disk(radius)
#         return kernel
#     for row in grid.T:
#         x, y = row
#         fake_image[x, y] = 1
#     kernel = new_kernel(radius)
#     fake_image = convolve2d(fake_image, kernel)
#     fake_image = cpx.interpolation.rotate(fake_image, theta)
#     return fake_image
def serious_pattern(kernel, xspacing, yspacing, xsize, ysize, rot_matrix):
    if abs(xspacing - yspacing) > 2:
        return None
    xspace = cp.arange(2, int(xsize), int(xspacing))
    yspace = cp.arange(2, int(ysize), int(yspacing))
    xs, ys = cp.meshgrid(yspace, xspace)
    grid = (cp.vstack([xs.ravel(), ys.ravel()]))
    grid = cp.dot(grid.T, rot_matrix).astype(cp.int32)
    grid += cp.absolute(grid.min(axis=0))
    maxs = int(grid.max()+16)
    fake_image = cp.zeros((maxs, maxs))
    fake_image[grid[:, 1], grid[:, 0]] = 1
    fake_image = cpx.binary_dilation(fake_image, structure=kernel, brute_force=True, )
    # fake_image = convolve2d(fake_image, kernel)
    return fake_image

def fit_pattern(corners, angle, angle_std, spacing, spacing_std, radius, binary_image, square_angle):
    vecx = corners[1, 0] - corners[0, 0]
    vecy = corners[1, 1] - corners[0, 1]
    spacing = cp.array(spacing)
    theta = cp.arccos((spacing * vecx) / (cp.sqrt(spacing ** 2) * cp.sqrt(vecx ** 2 + vecy ** 2)))
    rot_matrix = cp.array([[cp.cos(theta), -cp.sin(theta)], [cp.sin(theta), cp.cos(theta)]])
    square_angle2 = cp.rad2deg(theta).astype(cp.int32)

    def new_kernel(radius):
        kernel = cumorph.disk(radius)
        return kernel
    kernel = new_kernel(radius)
    print('hole radius is ', radius)
    if angle_std > 2:
        angle_sampling = cp.arange(square_angle2.get() - (2 * angle_std),
                                   square_angle2.get() + (2 * angle_std), step=2)
    else:
        angle_sampling = cp.arange(square_angle2.get()-(2*angle_std), square_angle2.get()+(2*angle_std), step=1)
    if spacing_std > 2:

        distance_sampling = cp.arange((spacing - 2*spacing_std).get(), (spacing + 2*spacing_std).get(), step=2)
    else:
        distance_sampling = cp.arange((spacing - 2 * spacing_std).get(), (spacing + 2 * spacing_std).get(), step=1)


    # print(f'angular sampling interval = {angle_sampling}')
    # print(f'spatial sampling interval = {distance_sampling}')
    def f(x):
        return cp.dot(rot_matrix, x)

    def get_size(array, clip=False):
        min = array.min()
        max = array.max()
        off = cp.array(0)
        if clip:
            if min < 0:
                off = min
                min = 0

            if max > 4096:
                off = max - cp.array(4096)
                max = 4096
            return max - min, off
        return max - min
    corners_flat = cp.apply_along_axis(f, 1, corners)
    size_x = get_size(corners_flat[:, 0])
    size_y = get_size(corners_flat[:, 1])
    size_x2, offx = get_size(corners[:, 0], clip=True)
    size_y2, offy = get_size(corners[:, 1], clip=True)
    # print(size_x2, size_y2, offx, offy)
    maxsize = cp.array([size_x, size_y]).max() * 2
    list = cp.array([[0, 0, 0, 0]])
    # try:
    for theta in angle_sampling:
        theta_rad = cp.deg2rad(theta)
        rot_matrix = cp.array([[cp.cos(theta_rad), -cp.sin(theta_rad)], [cp.sin(theta_rad), cp.cos(theta_rad)]])
        for xspacing in distance_sampling:
            for yspacing in distance_sampling:
                points = serious_pattern(kernel, xspacing, yspacing, maxsize, maxsize, rot_matrix)
                points = crop_center(points, size_x, size_y)
                points = cpx.rotate(points.astype(cp.float32), -square_angle2)
                points = cp.pad(points, 100)
                points = cp.fliplr(points)
                points = crop_center(points, size_x2, size_y2, offx=offx, offy=offy)
                # print(points.shape)
                value = align_points(binary_image, points, corners, size_x2, size_y2, xspacing, yspacing)
                list = cp.append(list, cp.array([[xspacing, yspacing, angle, value]]), axis=0)
    best_fit = list[cp.argmax(list[:, 3])]
    # except:
    #     print('square could not be aligned')
    #     return 0, 0, 0, 0
    if best_fit[0] == 0 and best_fit[2] == 0:
        print('square could not be aligned')
        return 0, 0, 0, 0
    else:
        print('final params: ', best_fit[0], best_fit[1], best_fit[2]-square_angle2)
        points = serious_pattern(radius, best_fit[0], best_fit[1], best_fit[2], maxsize, maxsize)
        # plt.imshow(points.get())
        # plt.show()
        points = crop_center(points, size_x, size_y)
        points = cpx.interpolation.rotate(points, -square_angle2)
        points = cp.pad(points, 100)
        points = cp.fliplr(points)
        points = crop_center(points, size_x2, size_y2, offx=offx, offy=offy)
        dataframe = align_points(binary_image, points, corners, size_x2, size_y2, spacing, spacing, final=True)
        return dataframe, (best_fit[2]-square_angle2).get(), best_fit[0].get(), best_fit[1].get()

def determine_angle(sample, dataframe, dot_dist):
    dot_dist = dot_dist.get()
    angles = []
    dists = []
    for idx, row in sample.iterrows():
        basic_holes = dataframe.copy()
        x = row['X']
        y = row['Y']
        basic_holes = basic_holes[basic_holes.X.values > x - (dot_dist+3)]
        basic_holes = basic_holes[basic_holes.X.values < x + (dot_dist+3)]
        basic_holes = basic_holes[basic_holes.Y.values > y - (dot_dist+3)]
        basic_holes = basic_holes[basic_holes.Y.values < y + (dot_dist+3)]
        basic_holes['dist'] = np.sqrt((basic_holes['X'] - x) ** 2 + (basic_holes['Y'] - y) ** 2)
        basic_holes = basic_holes[basic_holes.dist.values < (dot_dist+3)]
        basic_holes['X_v'] = basic_holes['X'] - x
        basic_holes['Y_v'] = basic_holes['Y'] - y
        basic_holes = basic_holes[basic_holes.X_v.values > 0]
        basic_holes = basic_holes[basic_holes.Y_v.values < 0]

        if len(basic_holes) > 0:
            basic_holes = basic_holes.reset_index()
            ix = basic_holes.first_valid_index()
            pos = basic_holes.iloc[ix]
            theta = np.arccos((dot_dist * pos['X_v']) / (np.sqrt(dot_dist ** 2) * np.sqrt(pos['X_v'] ** 2 + pos['Y_v'] ** 2)))
            angle = np.rad2deg(theta).astype(np.int32)
            angles.append(90 - angle)
            dists.append(pos['dist'])

    return angles, dists
def slice_tiles(image, offset_value, dataframe, dilation, radius, calc_means=False):
    def get_mask(dilation):
        mask = np.zeros((dilation*2, dilation*2))
        mask[draw.disk((dilation, dilation), radius=radius)] = 1
        return cp.array(mask)
    def get_slice(x, y, mask, image, dilation):
        x_1 = int(round(x - dilation))
        x_2 = int(round(x + dilation))
        y_1 = int(round(y - dilation))
        y_2 = int(round(y + dilation))

        slice = image[y_1:y_2, x_1:x_2]
        slice = slice * mask
        slice = slice[..., tf.newaxis]

        if slice.shape == (32, 32, 1):
            return slice.get()
        else:
            slice = cp.zeros((dilation * 2, dilation * 2, 1)).astype(cp.float32)
            assert slice.shape == (32, 32, 1)
            return slice.get()
    mask = get_mask(dilation)
    image = image + (255 - offset_value)
    image = image * (255 / image.max())

    list_of_slices = dataframe.apply(
        lambda x: get_slice(x.X, x.Y, mask, image, dilation), axis=1)

    def mean(slice):
        slice = slice[np.nonzero(slice)]
        mean = np.mean(slice)
        if mean > 0:
            return mean
        else:
            return 0
    if calc_means:
        list_of_means = [mean(x) for x in list_of_slices]
        return list_of_slices, list_of_means
    else:
        return list_of_slices
def return_dataset(data):
    dataset = tf.convert_to_tensor(data)
    dataset = tf.transpose(dataset, [2, 0, 1, 3])
    dataset = (dataset - 13.22129271735691) / 49.73459901227947 ##The mean and std of the training dataset
    return dataset
def get_dataset(image, offset_value, master, dilation, radius, thres):

    list_of_slices, means = slice_tiles(image, offset_value, master, dilation=dilation, radius=radius, calc_means=True)
    list_of_slices = np.stack(list_of_slices, axis=2)
    dataset_x = return_dataset(list_of_slices)
    return dataset_x, means
def calculate_thickness(tile, Grid=None): ## Intended to apply to list at the the end

    try:
        ## Step 1: binarize image and get initial guess for hole locations and pattern
        Tile = start_Tile_class(Grid, tile)
        print(f'{Tile.filename[-8:-4]} starting')
        size, size_std = Tile.get_hole_size()
        target_size = round(size / 2)
        dot_size, dot_dist, list_cent, blob, empties = cupy_calc_size_distance(Tile.image, 70, target_size)
        if len(list_cent) <= 100:
            pass
        else:
            Tile.binary_image = blob
            # plt.imshow(Tile.binary_image.get())
            # plt.show()
            Grid.add_hole_size(dot_size.get())
            Grid.add_hole_spacing(dot_dist.get())
            if len(empties) > 0:
                Grid.add_blanks(np.mean(empties))
                Tile.add_local_blanks(empties)
            Tile.coordinates = pd.DataFrame(data=list_cent.get(), columns=['Y', 'X'])

            ## Loose way to initialize some values to avoid combinatorically hard search
            angles, dists = determine_angle(Tile.coordinates.sample(10), Tile.coordinates, dot_dist)
            Grid.add_hole_angle(np.mean(angles))
            Grid.add_hole_angle(np.mean(angles))
            Grid.add_hole_angle(np.mean(angles) - 2*np.std(angles))
            Grid.add_hole_angle(np.mean(angles) + 2*np.std(angles))
            Grid.add_hole_spacing(np.mean(dists))
            Grid.add_hole_spacing(np.mean(dists))
            Grid.add_hole_spacing(np.mean(dists) - 2*np.std(dists))
            Grid.add_hole_spacing(np.mean(dists) + 2*np.std(dists))
            ## Step 2: determine location and angle of gridsquares, then fill in any holes missing from step 1
            square_mask, square_outlines, angles = find_gridsquares(Tile.image, 20)
            Tile.add_square_angle(angles[0])
            def check_for_overlap(x,y):
                return mask[round(y), round(x)] > 0
            mask = Tile.binary_image.get()
            mask = cpx.binary_dilation(Tile.binary_image, iterations=5, brute_force=True).get()
            for idx, square in enumerate(square_outlines):
                spacing, spacing_std = Grid.get_hole_spacing()
                angle, angle_std = Grid.get_hole_angle()
                dataframe,  angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                               spacing, spacing_std, Tile.get_radius(), Tile.binary_image, Tile.get_square_angle())


                if angle == 0:
                    pass
                else:
                    Grid.add_hole_angle(angle)
                    Grid.add_hole_spacing(spacing)
                    Grid.add_hole_spacing(spacingy)
                    dataframe['overlap'] = dataframe.apply(lambda x: check_for_overlap(x.X, x.Y), axis=1)
                    dataframe = dataframe[dataframe.overlap.values == False]
                    dataframe = dataframe[['X', 'Y']]
                    Tile.coordinates = pd.concat([Tile.coordinates, dataframe])
            # plt.imshow(Tile.image.get())
            # plt.scatter(x=Tile.coordinates['X'], y=Tile.coordinates['Y'])
            # plt.show()
            Tile.coordinates = Tile.coordinates[Tile.coordinates.X.values > 16]
            Tile.coordinates = Tile.coordinates[Tile.coordinates.Y.values > 16]
            Tile.coordinates = Tile.coordinates[Tile.coordinates.X.values < 4080]
            Tile.coordinates = Tile.coordinates[Tile.coordinates.Y.values < 4080]
            print(f'{Grid.name}/{Tile.filename[-8:-4]}.csv == {len(Tile.coordinates)} holes')
            Tile.coordinates.to_csv(f'{Grid.name}/{Tile.filename[-8:-4]}.csv')
            return Tile
    except:
        error = f'{tile[-8:-4]} failed silently'
        return error
    # print(dataset)

    # for_df = model.predict(dataset)
    # Tile.coordinates['CNN_prediction'] = for_df
    # COLOR = 'white'
    # cm.rcParams['text.color'] = COLOR
    # cm.rcParams['axes.labelcolor'] = COLOR
    # cm.rcParams['xtick.color'] = COLOR
    # cm.rcParams['ytick.color'] = COLOR
    # fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    # ax.imshow(Tile.image.get(), cmap='gray')
    # ax.set_title(f'Tile 0_1')
    # ax.set_facecolor('xkcd:black')
    # ax.scatter(x=Tile.coordinates['X'], y=Tile.coordinates['Y'], c=Tile.coordinates['CNN_prediction'], cmap='inferno_r', vmin=0, vmax=250,
    #            alpha=0.75, s=1)
    # plt.savefig(f'CNN_{Grid.name[0]}{tile[0]}.png', dpi=300)
    # plt.close()
    # Tile.coordinates.to_csv(f'CNN_{Grid.name[0]}{Tile.filename[-8:-4]}.csv')
def regression_statsmodels(dataframe, filename, thres):
  poly = np.polynomial.polynomial.polyfit(dataframe['mean'], dataframe['Ice_thickness_offset'], 3)

  y_pred = np.polynomial.polynomial.polyval(dataframe['mean'], poly)
  dataframe['y_pred'] = y_pred
  ### linear regression model
  # rlm = sm.RLM(dataframe[['Ice_thickness_offset']], sm.add_constant(dataframe[['means']]), M=sm.robust.norms.HuberT())
  # rlm_results = rlm.fit()
  # print(rlm_results.summary())
  # line_X = np.arange(np.min(dataframe['means']), np.max(dataframe['means']), 0.2).reshape(-1, 1)
  # line_y_rlm = rlm_results.predict(sm.add_constant(line_X))
  # y_pred = rlm_results.predict(sm.add_constant(dataframe['means']))

  ### polynomial model
  list_squares = dataframe.GridSquare.unique()
  idx = 0
  for name in list_squares:
    dataframe = dataframe.replace(to_replace=name, value=idx)
    idx+=1
  dataframe.plot.scatter(x='y_pred', y='Ice_thickness_offset', c='GridSquare', cmap='Set3', s=5)
  poly = np.polynomial.polynomial.polyfit(dataframe['y_pred'], dataframe['Ice_thickness_offset'], 3)
  x_new = np.linspace(0, 300, 300)
  y_pred = np.polynomial.polynomial.polyval(dataframe['y_pred'], poly)
  fit = np.polynomial.polynomial.polyval(x_new, poly)
  residuals = np.mean(abs(dataframe['Ice_thickness_offset'] - y_pred))
  correlation_matrix = np.corrcoef(dataframe['Ice_thickness_offset'], y_pred)
  correlation_xy = correlation_matrix[0, 1]
  r_squared = correlation_xy ** 2
  plt.title(f'average error = ∓ {residuals}  r_squared = {r_squared}')
  # plt.plot(x_new, fit, c="black", label="3rd order polynomial")
  # plt.plot(line_X, line_y_rlm, c="black", label="statsmodels_regression")
  plt.legend(loc='lower right')
  plt.savefig(f'poly3_{filename}_{thres}.png')
  return y_pred
def visualize_dataset(dataset):
    """
    Example of how to load dataset.pkl into its components, and some visualization of what the inside looks like
    """
    plt.rcParams.update({'font.size': 6})
    for ix, x in enumerate(dataset.numpy()):
        example = dataset.numpy()[ix]
        plt.imshow(example)
        plt.show()
def Get_dataset(model=None, predict=False):
    list_of_radius = np.array([6,8,10,12,14,16])
    master = pd.read_csv('offset_corrected_no_outliers.csv')
    master = master[master.corr_atlas_x.values > np.max(list_of_radius)]
    master = master[master.corr_atlas_x.values < 4096 - np.max(list_of_radius)]
    master = master[master.corr_atlas_y.values > np.max(list_of_radius)]
    master = master[master.corr_atlas_y.values < 4096 - np.max(list_of_radius)]
    list_of_grids = master.grid_class.unique()
    print(master.Tile_name.unique())
    global_dataset = Dataset([], [], None, None, None)
    for grid in list_of_grids:
        Grid = start_Grid_class()
        Grid.name = grid
        global_dataset.add_grid(grid)
        grid_df = master[master.grid_class.values == grid]
        print(grid)
        list_of_tiles = grid_df.Tile_name.unique()
        print(list_of_tiles)
        for tile in list_of_tiles:

            Tile = start_Tile_class(Grid, tile)
            tile_df = grid_df[grid_df.Tile_name.values == tile]
            Tile.coordinates = tile_df[['corr_atlas_x', 'corr_atlas_y']]
            Tile.ice_thickness = tile_df['Ice_thickness_offset']

            Tile.coordinates = Tile.coordinates.rename(columns={"corr_atlas_x": "X", "corr_atlas_y":"Y"})
            size, size_std = Tile.get_hole_size()
            target_size = round(size / 2)
            dot_size, dot_dist, list_cent, blob, empties = cupy_calc_size_distance(Tile.image, 70, target_size, special=True)
            if np.mean(empties) > 0:
                Grid.add_blanks(np.mean(empties))
                Tile.add_local_blanks(empties)
            for radius in list_of_radius:
                dataset, means = get_dataset(Tile.image, Tile.get_offset(Grid), Tile.coordinates, 16, radius, 90)
                tile_df['gray_means'] = means
                tile_df.to_csv(f'test_files/no_predict{Grid.name}{Tile.filename[-8:-4]}radius{radius}.csv')
                if predict:

                    answers = model.predict(dataset)
                    tile_df['CNN_prediction'] = answers
                    tile_df.to_csv(f'test_files/mult_{Grid.name}{Tile.filename[-8:-4]}radius{radius}.csv')
                global_dataset.add_doses(tile_df['DoseOnCamera_x'])
                global_dataset.add_dataset(dataset)
                # visualize_dataset(dataset)
    # print('finished', global_dataset.shape)
    return global_dataset

def run_CNN(train_x, test_x, train_y, test_y, dilation, name, btch):
    model = CNN_model_ala_Luke(dilation)
    print(model.summary())
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics='MeanAbsoluteError')
    log_path = f'tmp/atlas_model_radmultiple_preproc'
    callback = keras.callbacks.ModelCheckpoint(log_path, monitor='val_mean_absolute_error', save_best_only=True)
    model.fit(x=train_x, y=train_y, batch_size=btch, verbose=1, epochs=500, validation_data=(test_x, test_y), callbacks=callback)
    model.save(f'atlas_model_radiusmultiple_withpreproc')
    return model
def CNN_model_ala_Luke(name):
  inputs = keras.layers.Input((name*2, name*2, 1))
  rotation = keras.layers.RandomRotation(factor=(0, 1))(inputs)
  translation = keras.layers.RandomTranslation((-0.1, 0.1), (-0.1, 0.1))(rotation)
  x1 = keras.layers.Conv2D(filters=nfilters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(translation)
  x2 = keras.layers.Conv2D(filters=nfilters, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(x1)
  x3 = keras.layers.MaxPooling2D(3)(x2)
  x4 = keras.layers.Conv2D(filters=nfilters*2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(x3)
  x5 = keras.layers.Conv2D(filters=nfilters*2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(x4)
  x6 = keras.layers.MaxPooling2D(3)(x5)
  x7 = keras.layers.Flatten()(x6)
  x8 = keras.layers.Dropout(0.2)(x7)
  x9 = keras.layers.Dense(1, activation='linear')(x8)
  return keras.Model(inputs=inputs, outputs=x9)
def return_shuffled_dataset(data, shuffled_idx, test_split=25,shuffle=True, standardize=False, split=True):
    dataset = tf.convert_to_tensor(data)
    if standardize:
        mean = tf.math.reduce_mean(dataset)
        print('mean of training set is', mean)
        stddev = tf.math.reduce_std(dataset)
        print('std of training set is', stddev)
        dataset = (dataset - mean) / stddev
    if shuffle:
        dataset = tf.gather(dataset, shuffled_idx, axis=0)
    if split:
        test_set = dataset[:test_split]
        train_set = dataset[test_split:]
        return train_set, test_set
    if not split:
        return dataset
def Train(global_dataset):
    data_labels = global_dataset.doses
    data_labels = (data_labels-data_labels.mean())/data_labels.std()
    perm = tf.random.shuffle(tf.range(tf.shape(data_labels)[0]))
    data_size = int(len(data_labels)*0.25)
    train_x, test_x = return_shuffled_dataset(global_dataset.datasets, perm, test_split=data_size)
    train_y, test_y = return_shuffled_dataset(data_labels, perm, test_split=data_size)

    # to save it
    with open("dataset.pkl", "wb") as f:
        pkl.dump([train_x, test_x, train_y, test_y], f)
    model = run_CNN(train_x, test_x, train_y, test_y, 16, 5, 32)
import pickle as pkl

def Test():
    model = keras.models.load_model(
        f'/camp/home/shared/Machine_Learning_Datasets/All_datasets_with_atlas_alignment/tmp/atlas_model_radmultiple_preproc')
    Get_dataset(model=model, predict=True)
    # for_df = model.predict(train_x)
    # for_df2 = model.predict(test_x)
    # for_df = -870.8 * for_df + 649.1
    # for_df2 = -870.8 * for_df2 + 649.1
    # test_y = -870.8 * test_y + 649.
    # train_y = -870.8 * train_y + 649.1
    # plt.scatter(for_df, train_y)
    # plt.scatter(for_df2, test_y)
    # plt.show()
    # model = CNN_model_ala_Luke(16)
    # model.compile(optimizer='adam', loss='mean_absolute_error', metrics='MeanAbsoluteError')
    # model.load_weights(f'tmp/box16_mask532')
    print(model.summary())

# def conductor(file):
#     parent_dir = os.getcwd()
#     name = f'{file[0:-4]}_inference'
#     path = os.path.join(parent_dir, name)
#     if not os.path.exists(path):
#         os.mkdir(path)
#     Grid = start_Grid_class(file, path)
#     def worker(list):
#         for item in list:
#             calculate_thickness(Grid, item)
#     list_tiles = Grid.get_list_of_tiles()
#     procs = []
#     for i in range(3):
#         p = mp.Process(target=worker, args=list_tiles)
#         procs.append(p)
#         p.start()
#     for p in procs:
#         p.join()


# all_grids = glob.glob('W:/home/shared/Machine_Learning_Datasets/Inference/*atlas.csv')

# for grid in all_grids:
#     parent_dir = os.getcwd()
#     name = f'{grid[0:-4]}_inference'
#     path = os.path.join(parent_dir, name)
#     if not os.path.exists(path):
#         os.mkdir(path)
#     Grid = start_Grid_class(grid, path)
#     list_tiles = Grid.get_list_of_tiles()
#     for item in list_tiles:
#         calculate_thickness(item, Grid=Grid)
# if args.mode == 'training':
#     global_dataset = Get_dataset()
#     # visualize_dataset(train_x)
#     Train(global_dataset)
#
#
# if args.mode == 'test':
#     Test()

# if args.mode == 'inference':
#     Grid = start_Grid_class()
#     list_tiles = Grid.get_list_of_tiles()
    # if __name__ == '__main__':
    #     pool = mp.Pool(args.threads)
    #     list=[]
    #     for item in list_tiles:
    #         print(os.path.exists(f'{Grid.name}{item[-8:-4]}.csv'))
    #         if os.path.exists(f'{Grid.name}{item[-8:-4]}.csv'):
    #             pass
    #         else:
    #             list.append(item)
    #     print(list)
    #     for item in list:
    #         result = pool.apply_async(calculate_thickness, args=(item, ))
    #     pool.close()
    #     pool.join()
    #     pool = mp.Pool(args.threads)
    #     list_missed = []
    #     for item in list_tiles:
    #         if os.path.exists(f'{Grid.name}{item[-8:-4]}.csv'):
    #             pass
    #         else:
    #             list_missed.append(item)
    #     print(list_missed)
    #     for item in list_missed:
    #         result = pool.apply_async(calculate_thickness, args=(item, ))
    #     pool.close()
    #     pool.join()
    # list=[]
    # for item in list_tiles:
    #     print(os.path.exists(f'{Grid.name}{item[-8:-4]}.csv'))
    #     if os.path.exists(f'{Grid.name}{item[-8:-4]}.csv'):
    #         pass
    #     else:
    #         list.append(item)
    # for item in list:
    #     result = calculate_thickness(item)
    #     print(result)#single threaded inference