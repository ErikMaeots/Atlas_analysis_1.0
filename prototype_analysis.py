import argparse
from Master_load_metadata_from_collection import make_csv
from Grid_analysis_library import *
import cupy as cp
import cupyx.scipy.ndimage as cpx
import matplotlib.pyplot as plt
from cucim.skimage.morphology import disk
import itertools
import numpy as np
import pandas as pd
from datetime import datetime
from hex_pattern import serious_hex_pattern
from multiprocessing_attempty import Grid, Tile, start_Tile_class
from inference_test import final_analysis
from tqdm import tqdm


mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()
parser = argparse.ArgumentParser(
    description='Returns Ice thickness estimates based on model atlas_model_radmultiple_preproc')
parser.add_argument("-input", help='path to atlas tiles',
                    default='w:/home/shared/Machine_Learning_Datasets/gNE0018/Atlas/')
parser.add_argument("-input2", help='optional path to ground truth dose measurements')
parser.add_argument("-output", help='folder for outputs',
                    default='w:/home/shared/Machine_Learning_Datasets/Inference/')
parser.add_argument("-name", help='filename prefix to associate with results', default='gNEXXX')
parser.add_argument("-mode", help='use "hex" for hex grids', default='inference')
parser.add_argument("-target_size", help='hole size in pixel area, default is usually good, for hex grids use 5')
args = parser.parse_args()

def start_Grid_class(meta_file, output):
    meta_file = apply_regex_to_df(meta_file, '.xml', '.mrc')
    # meta_file = apply_regex_to_df(meta_file, '/camp', 'W:')
    # meta_file = apply_regex_to_df(meta_file, '/', r"\'")
    grid = output
    return Grid(grid, meta_file, [], [], [], [], [])


def new_kernel(radius):
    kernel = disk(radius)
    return kernel


def fit_pattern(corners, angle, angle_std, spacing, spacing_std, radius, binary_image):
    vecx = corners[1, 0] - corners[0, 0]
    vecy = corners[1, 1] - corners[0, 1]
    # plt.imshow(binary_image.get())
    # plt.show()
    spacing = cp.array(spacing)
    theta = cp.arccos((spacing * vecx) / (cp.sqrt(spacing ** 2) * cp.sqrt(vecx ** 2 + vecy ** 2)))
    rot_matrix = cp.array([[cp.cos(theta), -cp.sin(theta)], [cp.sin(theta), cp.cos(theta)]])
    square_angle2 = cp.rad2deg(theta).astype(cp.int32)
    kernel = disk(radius)
    # print('hole radius is ', radius)
    if angle_std > 2:
        angle_sampling = cp.arange(square_angle2.get() + angle - (2 * angle_std),
                                   square_angle2.get() + angle + (2 * angle_std), step=2)
    else:
        angle_sampling = cp.arange(square_angle2.get() + angle - (2 * angle_std),
                                   square_angle2.get() + angle + (2 * angle_std), step=1)
    if spacing_std > 2:
        distance_sampling = cp.arange((spacing - 2 * spacing_std).get(), (spacing + 2 * spacing_std).get(), step=2)
    else:
        distance_sampling = cp.arange((spacing - 2 * spacing_std).get(), (spacing + 2 * spacing_std).get(), step=1)

    print(f'angular sampling interval = {angle_sampling}')
    print(f'spatial sampling interval = {distance_sampling}')
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

            if max > 4095:
                off = max - cp.array(4095)
                max = 4095
            return max - min, off
        return max - min
    corners_flat = cp.apply_along_axis(f, 1, corners)
    size_x = get_size(corners_flat[:, 0])
    size_y = get_size(corners_flat[:, 1])
    size_x2, offx = get_size(corners[:, 0], clip=True)
    size_y2, offy = get_size(corners[:, 1], clip=True)
    # print(size_x2, size_y2, offx, offy)
    maxsize = cp.array([size_x, size_y]).max() * 2

    def serious_pattern(kernel, comb):
        xspace, yspace, rot_matrix = comb
        if args.mode == 'hex':
            grid = serious_hex_pattern(xspace.min().get(), yspace.min().get(), xspace.max().get(), yspace.max().get(), (xspace[1] - xspace[0]).get())
            # plt.scatter(x=grid[:, 0].get(), y=grid[:, 1].get())
            # plt.show()
            grid = cp.dot(grid, rot_matrix).astype(cp.int32)
            grid += cp.absolute(grid.min(axis=0))

        else:
            if abs((xspace[1] - xspace[0]) - (yspace[1] - yspace[0])) > 2:  # Don't search skewed grids
                return None
            xs, ys = cp.meshgrid(yspace, xspace)
            grid = (cp.vstack([xs.ravel(), ys.ravel()]))
            grid = cp.dot(grid.T, rot_matrix).astype(cp.int32)
            grid += cp.absolute(grid.min(axis=0))
        maxs = int(grid.max() + 16)
        fake_image = cp.zeros((maxs, maxs))
        fake_image[grid[:, 1], grid[:, 0]] = 1
        fake_image = cpx.binary_dilation(fake_image, structure=kernel, brute_force=True, )
        # fake_image = convolve2d(fake_image, kernel) ##for some reason binary_dilation with the same kernel is faster
        points = crop_center(fake_image, size_x, size_y)
        points = cpx.rotate(points.astype(cp.float32), -square_angle2)
        points = cp.pad(points, 100)
        points = cp.fliplr(points)
        points = crop_center(points, size_x2, size_y2, offx=offx, offy=offy)
        return points
    try:
        distances = [cp.arange(2, int(maxsize), int(x)) for x in distance_sampling]
    except:
        print(' FAILED with NaN skipping square')
        return 0, 0, 0, 0
    angles_rad = [cp.deg2rad(theta) for theta in angle_sampling]
    angles = [cp.array([[cp.cos(theta_rad), -cp.sin(theta_rad)], [cp.sin(theta_rad), cp.cos(theta_rad)]]) for
              theta_rad in angles_rad]
    combinations = list(itertools.product(distances, distances, angles))
    print(' Combinatoric complexity = ', len(combinations), 'permutations')
    combinations_inp = list(itertools.product(distance_sampling, distance_sampling, angle_sampling))
    all_images = [serious_pattern(kernel, comb) for comb in combinations]
    final_list = list(zip(combinations, all_images))
    alignments = [align_points(binary_image, points, corners, size_x2, size_y2,
                               (spacing + 2 * spacing_std),
                               (spacing + 2 * spacing_std)) for points in all_images]

    # alignments = [align(all_images[9], x) for x in final_list]
    winning_index = cp.argmax(cp.vstack(alignments)).get()

    if alignments[winning_index] == 0:
        print('square could not be aligned')
        return 0, 0, 0, 0
    points = serious_pattern(kernel, combinations[winning_index])
    winning_xspace, winning_yspace, winning_angle = combinations_inp[winning_index]
    winning_angle =(winning_angle-square_angle2).get()
    print('final params: ', winning_angle, winning_xspace, winning_yspace)
    dataframe = align_points(binary_image, points, corners, size_x2, size_y2, winning_xspace, winning_xspace, final=True)
    return dataframe, winning_angle, winning_xspace.get(), winning_yspace.get()



def process_tile(grid_class, tile_path):
    tile_class = start_Tile_class(grid_class, tile_path)
    if os.path.exists(f'{grid_class.name}/{tile_class.filename[-8:-4]}.csv'):
        tile_class.coordinates = pd.read_csv(f'{grid_class.name}/{tile_class.filename[-8:-4]}.csv')
        tile_class.local_blanks = np.load(f'{path}/{tile_class.filename[-8:-4]}_blanks.npy')
        tile_class.image = load_image(tile_path)
        return tile_class
    size, size_std = grid_class.get_hole_size()
    if args.target_size:
        target_size = int(args.target_size)
    else:
        target_size = round(size / 2)
    dot_size, dot_dist, list_cent, blob, empties = cupy_calc_size_distance(tile_class.image, 70, target_size)
    if len(list_cent) <= 100:
        print(f"Oops, {tile_path} didn't look promising, skipped to prevent crashing")
        return None
    else:
        print(f"Launched {tile_path}")
    tile_class.binary_image = blob
    grid_class.add_hole_size(dot_size.get())
    grid_class.add_hole_spacing(dot_dist.get())
    if len(empties) > 0:
        grid_class.add_blanks(np.mean(empties)) ##save out broken areas to use for normalization later
        tile_class.add_local_blanks(empties) ##local copy of the same
    tile_class.coordinates = pd.DataFrame(data=list_cent.get(), columns=['Y', 'X'])
    if len(grid_class.hole_angle) > 0: ##Only run the below initialize
        pass
    else:
        angles, dists = determine_angle(tile_class.coordinates.sample(100), tile_class.coordinates, dot_dist)
        if np.std(angles) > 3:
            return None
        elif np.std(dists) > 3:
            return None
        else:
            grid_class.add_hole_angle(np.mean(angles))
            grid_class.add_hole_angle(np.mean(angles))
            grid_class.add_hole_angle(np.mean(angles) - 2 * np.std(angles))
            grid_class.add_hole_angle(np.mean(angles) + 2 * np.std(angles))
            grid_class.add_hole_spacing(np.mean(dists))
            grid_class.add_hole_spacing(np.mean(dists))
            grid_class.add_hole_spacing(np.mean(dists) - 2 * np.std(dists))
            grid_class.add_hole_spacing(np.mean(dists) + 2 * np.std(dists))
    square_mask, square_outlines, angles = find_gridsquares(tile_class.image, 20, args.mode)
    tile_class.add_square_angle(cp.median(cp.asarray(angles)).get())

    def check_for_overlap(x, y):
        return mask[round(y), round(x)] > 0

    mask = tile_class.binary_image.get()
    mask = cpx.binary_dilation(tile_class.binary_image, iterations=5, brute_force=True).get()
    if args.mode == 'hex':
        tile_class.coordinates = tile_class.coordinates.iloc[0]
    for idx, square in enumerate(tqdm(square_outlines)):
        # print(f'{tile_class.filename}', idx, f'/{len(square_outlines)}') ## progress count
        spacing, spacing_std = grid_class.get_hole_spacing(args.mode)
        angle, angle_std = grid_class.get_hole_angle(args.mode)
        print(args.mode)
        if args.mode == 'hex':
            dataframe, angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                                              spacing, spacing_std, tile_class.get_radius(args.mode),
                                                              (tile_class.image / tile_class.image.max()),
                                                              )
        else:
            if len(list_cent) <= 600:
                dataframe, angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                                                  spacing, spacing_std, tile_class.get_radius(args.mode),
                                                                  (tile_class.image / tile_class.image.max()),
                                                                  )
            else:
                dataframe, angle, spacing, spacingy = fit_pattern(square, angle, angle_std,
                                                                  spacing, spacing_std, tile_class.get_radius(args.mode),
                                                                  tile_class.binary_image,
                                                                  )

        if angle == 0:
            pass
        else:
            grid_class.add_hole_angle(angle)
            grid_class.add_hole_spacing(spacing)
            grid_class.add_hole_spacing(spacingy)
            if args.mode != 'hex':
                dataframe['overlap'] = dataframe.apply(lambda x: check_for_overlap(x.X, x.Y), axis=1)
                dataframe = dataframe[dataframe.overlap.values == False]
            dataframe = dataframe[['X', 'Y']]
            tile_class.coordinates = pd.concat([tile_class.coordinates, dataframe])


    # plt.imshow(tile_class.image.get(), cmap='gray')
    # plt.scatter(x=tile_class.coordinates['X'], y=tile_class.coordinates['Y'], s=0.1, alpha=0.2)
    # plt.savefig(f'{grid_class.name}/{tile_class.filename[-8:-4]}.png', dpi=600)
    # plt.close()

    tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.X.values > 16]
    tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.Y.values > 16]
    tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.X.values < 4080]
    tile_class.coordinates = tile_class.coordinates[tile_class.coordinates.Y.values < 4080]
    print(f'{path}/{tile_class.filename[-8:-4]}.csv == {len(tile_class.coordinates)} holes')
    tile_class.coordinates.to_csv(f'{grid_class.name}/{tile_class.filename[-8:-4]}.csv')
    np.save(f'{path}/{tile_class.filename[-8:-4]}_blanks.npy', tile_class.local_blanks)
    #lazy way to clear unnecessary parts of the class.
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    tile_class.name = None
    tile_class.meta_file = None
    tile_class.hole_size = None
    tile_class.hole_spacing = None
    tile_class.square_angle = None
    tile_class.hole_angle = None
    tile_class.binary_image = None
    tile_class.doses = None
    tile_class.ice_thickness = None
    return tile_class
def preprocess_grid(atlasDATA):
    grid_class = start_Grid_class(atlasDATA, path)
    list_tiles = grid_class.get_list_of_tiles()
    fitted_classes = [process_tile(grid_class, tile) for tile in list_tiles]
    model = keras.models.load_model(
        f'/camp/home/shared/Machine_Learning_Datasets/All_datasets_with_atlas_alignment/tmp/atlas_model_radmultiple_preproc')
    print(model.summary())
    # inter_layers4 = model.layers[:4]
    # inter_layers = model.layers[:5]
    # from tensorflow.keras.models import Sequential
    # inter_model = Sequential(inter_layers)
    # inter_model4 = Sequential(inter_layers4)
    for tile_class in fitted_classes:
        if tile_class is not None:
            blanks = tile_class.local_blanks
            if len(blanks) == 0:
                blanks = grid_class.blanks
            slices, means = get_dataset(tile_class.image, np.median(np.array(blanks).flatten()), tile_class.coordinates, 16, 16, 20)
            for_df = model.predict(slices)
            # for idx, slice in enumerate(slices):
            #     print('slice', idx)
            #     plt.imshow(slice, cmap='gray')
            #     plt.savefig('raw_image.png')
            #     plt.close()
            #     example = inter_model4.predict(np.expand_dims(slice, 0))
            #
            #     fig, ax = plt.subplots(4, 4)
            #     for idx, ax_idx in enumerate(list(itertools.product(range(4), range(4)))):
            #         ax[ax_idx[1], ax_idx[0]].imshow(example[0, :, :, idx])
            #     plt.savefig('computer_vision.png')
            #     example = inter_model.predict(np.expand_dims(slice, 0))
            #     fig2, ax2 = plt.subplots(4, 4)
            #     for idx, ax_idx in enumerate(list(itertools.product(range(4), range(4)))):
            #         ax2[ax_idx[1], ax_idx[0]].imshow(example[0, :, :, idx])
            #     plt.show()
            #     plt.savefig('computer_vision.png')

            tile_class.coordinates['mean'] = means
            tile_class.coordinates['CNN_prediction'] = for_df
            tile_class.coordinates['ice'] = (12.5 * (tile_class.coordinates['CNN_prediction'] ** 2)) + (-42.47 * tile_class.coordinates['CNN_prediction']) + 32.43
            # plt.scatter(x=tile_class.coordinates['X'], y=tile_class.coordinates['Y'], c=tile_class.coordinates['ice'], vmin=0, vmax=200, cmap='inferno_r', alpha=0.5,
            #             s=1)
            # plt.imshow(tile_class.image.get(), cmap='gray')
            # plt.show()
            tile_class.coordinates.loc[tile_class.coordinates['mean'] < 70, 'ice'] = 250
            # dot_dist, dot_std = grid_class.get_hole_spacing(args.mode)
            dot_dist = 19
            tile_class.coordinates = final_analysis(tile_class.coordinates, dot_dist)
            tile_class.coordinates.loc[tile_class.coordinates['neighbour'] == False, 'ice'] = 0

            if args.mode == 'hex':
                tile_class.coordinates.loc[tile_class.coordinates['mean'] < 65, 'ice'] = 250
            else:
                tile_class.coordinates.loc[tile_class.coordinates['mean'] < 70, 'ice'] = 250
            tile_class.coordinates.to_csv(f'{path}/{tile_class.filename[-8:-4]}_inference.csv')
            # fig, ax = plt.subplots(1,1,figsize=(8,8))
            plt.scatter(x=tile_class.coordinates['X'], y=tile_class.coordinates['Y'], c=tile_class.coordinates['ice'], vmin=0, vmax=200, cmap='inferno_r', alpha=0.5,
                        s=0.1)
            color_bar = plt.colorbar()
            color_bar.set_label('Predicted thickness (nm)')
            color_bar.set_alpha(1)
            color_bar.draw_all()
            plt.imshow(tile_class.image.get(), cmap='gray')
            plt.savefig(f'{path}/{tile_class.filename[-8:-4]}_inference_new.png', dpi=600)
            plt.close()

atlas_list = glob.glob(f'{args.input}Tile*.xml', recursive=False)
assert len(atlas_list) > 0, f'Oops, did not find any atlas tiles at {args.input}*.xml'
path = os.path.join(args.output, args.name)
if not os.path.exists(path):
    os.mkdir(path)
atlasDATA = make_csv(atlas_list, f'{path}_atlas.csv')
if args.input2:
    foilholes_list = glob.glob(f'{args.input2}/GridSquare*/FoilHoles/*.xml', recursive=True)
    highmag_list = glob.glob(f'{args.input2}/GridSquare*/Data/*.xml', recursive=True)
    gridsquare_list = glob.glob(f'{args.input2}/GridSquare*/*.xml', recursive=False)
    if len(foilholes_list) >= 1:
        foilDATA = make_csv(foilholes_list, f'{path}_foils.csv')
    if len(highmag_list) >= 1:
        holeDATA = make_csv(highmag_list, f'{path}_holes.csv')
    if len(gridsquare_list) >= 1:
        gridDATA = make_csv(gridsquare_list, f'{path}_squares.csv')
#

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

preprocess_grid(atlasDATA)
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)