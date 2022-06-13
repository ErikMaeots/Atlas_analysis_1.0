import multiprocessing as mp
import numpy as np
import pandas as pd
import cupy as cp
import mrcfile
import glob
import os
import cupyx.scipy.ndimage as cpx
import matplotlib.pyplot as plt
from datetime import datetime
assert cp.cuda.runtime.getDeviceCount() > 0, 'GPU not detected'
dev1 = cp.cuda.Device(0)
dev1.use()

from Grid_analysis_library import cupy_calc_size_distance, determine_angle, find_gridsquares, fit_pattern
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
    def get_radius(self, mode=None):
        if mode == 'hex':
            return 4
        radius = np.sqrt(np.mean(self.hole_size) / np.pi)
        if radius < 6:
            radius = 6
        elif radius > 16:
            radius = 16
        return radius
    def add_hole_spacing(self, spacing):
        self.hole_spacing.append(spacing)
    def get_hole_spacing(self, mode=None):
        if len(self.hole_spacing) > 3:
            if mode == 'hex':
                return 19, 0.5
            std = np.nanstd(self.hole_spacing)
            mean = np.nanmean(self.hole_spacing)
            if std < 0.5:
                std = 0.5
            elif std > 7.5:
                std = 7.5
            elif not 25 < mean < 55:
                mean = 40
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
    def get_hole_angle(self, mode):
        if len(self.hole_angle) > 3:

            std = np.std(self.hole_angle)
            if mode == 'hex':
                if std < 0.5:
                    std = 0.5
                if std > 15:
                    std = 15
                return np.mean(self.hole_angle), std
            if std < 0.5:
                std = 0.5
            if std > 22.5:
                std = 22.5
            return np.mean(self.hole_angle), std
        else:
            return 45, 22.5 ## defaults to 90 degrees spatial search
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
    meta_file = apply_regex_to_df(meta_file, 'W:', '/camp')
    # meta_file = apply_regex_to_df(meta_file, '/', r"\'")
    grid = output
    return Grid(grid, meta_file, [], [], [], [], [])
def load_image(filename, new_size=0, k=1):
  with mrcfile.open(filename) as f:
    f = cp.array(f.data)
    f = cp.rot90(f, k=k)
    f = cp.fliplr(f)
    f = f * (255 / f.max())
    return f.astype(cp.int32)

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




# def callback(result):
#     print(result)

def multiprocessing_function(grid):
    parent_dir = os.getcwd()
    name = f'{grid[0:-4]}_inference'
    path = os.path.join(parent_dir, name)
    if not os.path.exists(path):
        os.mkdir(path)
    grid_class = start_Grid_class(grid, path)
    list_tiles = grid_class.get_list_of_tiles()
    for idx, tile in enumerate(list_tiles):
        print(idx, tile)
        if idx > 60:
            break

        else:
            try:
                tile_class = start_Tile_class(grid_class, tile)
                if not os.path.exists(f'{grid_class.name}/{tile_class.filename[-8:-4]}_blanks.npy'):
                    size, size_std = tile_class.get_hole_size()
                    target_size = round(size / 2)
                    dot_size, dot_dist, list_cent, blob, empties = cupy_calc_size_distance(tile_class.image, 70, target_size)
                    if len(list_cent) <= 100:
                        pass
                    else:
                        tile_class.binary_image = blob
                        # plt.imshow(Tile.binary_image.get())
                        # plt.show()
                        grid_class.add_hole_size(dot_size.get())
                        grid_class.add_hole_spacing(dot_dist.get())
                        if len(empties) > 0:
                            grid_class.add_blanks(np.mean(empties))
                            tile_class.add_local_blanks(empties)
                        tile_class.coordinates = pd.DataFrame(data=list_cent.get(), columns=['Y', 'X'])

                        ## Loose way to initialize some values to avoid combinatorically hard search
                        if len(grid_class.hole_angle) > 0:
                            pass
                        else:

                            angles, dists = determine_angle(tile_class.coordinates.sample(100), tile_class.coordinates, dot_dist)
                            if np.std(angles) > 3:
                                list_tiles.append(pd.Series(tile))
                                pass
                            elif np.std(dists) > 3:
                                list_tiles.append(pd.Series(tile))
                                pass
                            else:
                                grid_class.add_hole_angle(np.mean(angles))
                                grid_class.add_hole_angle(np.mean(angles))
                                grid_class.add_hole_angle(np.mean(angles) - 2 * np.std(angles))
                                grid_class.add_hole_angle(np.mean(angles) + 2 * np.std(angles))
                                grid_class.add_hole_spacing(np.mean(dists))
                                grid_class.add_hole_spacing(np.mean(dists))
                                grid_class.add_hole_spacing(np.mean(dists) - 2 * np.std(dists))
                                grid_class.add_hole_spacing(np.mean(dists) + 2 * np.std(dists))
                        square_mask, square_outlines, angles = find_gridsquares(tile_class.image, 20)

                        tile_class.add_square_angle(angles[0])

                        def check_for_overlap(x, y):
                            return mask[round(y), round(x)] > 0

                        mask = tile_class.binary_image.get()
                        mask = cpx.binary_dilation(tile_class.binary_image, iterations=5, brute_force=True).get()
                        for idx, square in enumerate(square_outlines):
                            print(f'{tile_class.filename}', idx, f'/{len(square_outlines)}')
                            spacing, spacing_std = grid_class.get_hole_spacing()
                            angle, angle_std = grid_class.get_hole_angle()
                            try:
                                if len(list_cent) <= 600:
                                    dataframe, angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                                                                      spacing, spacing_std, tile_class.get_radius(),
                                                                                      (tile_class.image / tile_class.image.max()),
                                                                                      tile_class.get_square_angle(), dot_size)
                                else:
                                    dataframe, angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                                                                    spacing, spacing_std, tile_class.get_radius(),
                                                                                    tile_class.binary_image, tile_class.get_square_angle())

                                if angle == 0:
                                    pass
                                else:
                                    grid_class.add_hole_angle(angle)
                                    grid_class.add_hole_spacing(spacing)
                                    grid_class.add_hole_spacing(spacingy)
                                    dataframe['overlap'] = dataframe.apply(lambda x: check_for_overlap(x.X, x.Y), axis=1)
                                    dataframe = dataframe[dataframe.overlap.values == False]
                                    dataframe = dataframe[['X', 'Y']]
                                    tile_class.coordinates = pd.concat([tile_class.coordinates, dataframe])
                            except:
                                pass

                        plt.imshow(tile_class.image.get())
                        plt.scatter(x=tile_class.coordinates['X'], y=tile_class.coordinates['Y'], s=1, alpha=0.2)
                        plt.savefig(f'{grid_class.name}/{tile_class.filename[-8:-4]}.png', dpi=600)
                        plt.close()

                        tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.X.values > 16]
                        tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.Y.values > 16]
                        tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.X.values < 4080]
                        tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.Y.values < 4080]
                        print(f'{grid_class.name}/{tile_class.filename[-8:-4]}.csv == {len(tile_class.coordinates)} holes')
                        tile_class.coordinates.to_csv(f'{grid_class.name}/{tile_class.filename[-8:-4]}.csv')
                        np.save(f'{grid_class.name}/{tile_class.filename[-8:-4]}_blanks.npy', tile_class.local_blanks)
            except:
                pass
    return grid_class, list_tiles


# all_grids = glob.glob('/camp/home/shared/Machine_Learning_Datasets/Inference/*atlas.csv')
# print(all_grids)
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)
# for idx, grid in enumerate(all_grids):
#     print(f'{idx}/{len(all_grids)}  START: {grid}')
#     multiprocessing_function(grid)
#
# now = datetime.now()
# current_time = now.strftime("%H:%M:%S")
# print("Current Time =", current_time)