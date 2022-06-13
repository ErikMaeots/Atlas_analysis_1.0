import numpy as np
import glob
import re
import uuid
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from Master_load_metadata_from_collection import make_csv
import os
import multiprocessing as mp
from Grid_analysis_library import load_image, get_dataset, apply_regex_to_df
nr_threads = 3
import threading
import concurrent.futures
import traceback


## Generate _atlas files for all found grids
# atlases = glob.glob('**/Atlas/', recursive=True)
# for atlas in atlases:
#     atlas_list = glob.glob(f'{atlas}Tile*.xml', recursive=False)
#     name = re.search('gNE[0-9]+', f'{atlas[-14:-7]}')
#     if name is None:
#         name = uuid.uuid4().hex
#         print('Found grid', name, 'with', len(atlas_list), 'atlas tiles')
#         if len(atlas_list) > 0:
#             make_csv(atlas_list, f'W:/home/shared/Machine_Learning_Datasets/Inference/gNE_unk_{name}_atlas.csv')
#     else:
#         print('Found grid', name.group(0), 'with', len(atlas_list), 'atlas tiles')
#         if len(atlas_list) > 0:
#             make_csv(atlas_list, f'W:/home/shared/Machine_Learning_Datasets/Inference/{name.group(0)}_atlas.csv')

## Run hole finding on all.
# metadata = glob.glob('W:/home/shared/Machine_Learning_Datasets/Inference/*atlas.csv')
# all_grids = glob.glob('W:/home/shared/Machine_Learning_Datasets/Inference/*inference/')
def final_analysis(df, dot_dist):

    # df = df.sort_values(by=['CNN_prediction'])
    # histo = df['CNN_prediction']
    # hist, edges = np.histogram(histo, bins=256)
    # peaks, properties = find_peaks(hist, prominence=len(df)/57)
    # th_min = edges[np.min(peaks)]
    # th_max = edges[np.max(peaks)]
    # df = df[df.CNN_prediction.values > th_min + 10]
    # df = df[df.CNN_prediction.values < th_max - 10]
    # df['CNN_prediction'] = df['CNN_prediction'] - edges[np.min(peaks)] ##normalize predictions to 0
    def check_neighbours(x, y, df, dot_dist):
        df = df[df.ice.values > (df.ice.values.min()+5)]
        df = df[df.ice.values < 250]
        df = df[df.X.values < x + dot_dist + 5]
        df = df[df.X.values > x - dot_dist - 5]
        df = df[df.Y.values < y + dot_dist + 5]
        df = df[df.Y.values > y - dot_dist - 5]
        df['dist'] = np.sqrt((df['X'] - x) ** 2 + (df['Y'] - y) ** 2)
        df = df[df.dist.values < dot_dist + 5]
        return len(df) >= 3

    df['neighbour'] = df.apply(lambda x: check_neighbours(x.X, x.Y, df, dot_dist), axis=1)
    df.loc[df['neighbour'] == False, 'ice'] = 0
    # df = df[df.neighbour.values == True]
    return df

# for file in metadata:
#     model = keras.models.load_model(
#         f'W:/home/shared/Machine_Learning_Datasets/All_datasets_with_atlas_alignment/tmp/atlas_model_radmultiple_preproc')
#     parent_dir = os.getcwd()
#     name = f'{file[0:-4]}_inference'
#     path = os.path.join(parent_dir, name)
#     df = pd.read_csv(file)
#     df = apply_regex_to_df(df, '.xml', '.mrc')
#     df = apply_regex_to_df(df, '/camp', 'W:')
#     blanks_global = []
#     for tile in df['filename']:
#         # np.save(f'{file}/{tile_class.filename[-8:-4]}_blanks.npy', tile_class.local_blanks)
#         if os.path.exists(f'{path}/{tile[-8:-4]}.csv'):
#             if os.path.exists(f'{path}/{tile[-8:-4]}_inference_new.png'):
#                 print(f'{path}/{tile[-8:-4]}_inference_new.png already exists!')
#             else:
#                 print(f'{path}/{tile[-8:-4]}_inference.csv')
#                 coords = pd.read_csv(f'{path}/{tile[-8:-4]}.csv')
#                 blanks = np.load(f'{path}/{tile[-8:-4]}_blanks.npy')
#                 if len(blanks) == 0:
#                     if len(blanks_global) == 0:
#                         all_blanks = glob.glob(f'{path}/*_blanks.npy')
#                         for bln in all_blanks:
#                             array = np.load(bln)
#                             if len(array) > 0:
#                                 if len(blanks_global) == 0:
#                                     blanks_global = array
#                                 else:
#
#                                     blanks_global = np.concatenate((blanks_global, array), axis=1)
#                         blanks = blanks_global
#                     else:
#                         blanks = blanks_global
#                 assert len(blanks) > 0
#
#                 slices, means = get_dataset(load_image(tile), np.mean(np.array(blanks).flatten()), coords, 16, 16, 20)
#                 for_df = model.predict(slices)
#                 coords['mean'] = means
#                 coords['CNN_prediction'] = for_df
#                 coords['ice'] = (12.5 * (coords['CNN_prediction'] ** 2)) + (-42.47 * coords['CNN_prediction']) + 32.43
#
#                 coords.loc[coords['mean'] < 70, 'ice'] = 250
#                 coords = final_analysis(coords)
#                 coords.loc[coords['neighbour'] == False, 'ice'] = 0
#                 coords.loc[coords['mean'] < 70, 'ice'] = 250
#                 coords.to_csv(f'{path}/{tile[-8:-4]}_inference.csv')
#                 image = load_image(tile)
#                 results = coords
#                 # fig, ax = plt.subplots(1,1,figsize=(8,8))
#                 plt.scatter(x=results['X'], y=results['Y'], c=results['ice'], vmin=0, vmax=200, cmap='inferno_r', alpha=0.5, s=1)
#                 color_bar = plt.colorbar()
#                 color_bar.set_label('Predicted thickness (nm)')
#                 color_bar.set_alpha(1)
#                 color_bar.draw_all()
#                 plt.imshow(image.get(), cmap='gray')
#                 plt.savefig(f'{path}/{tile[-8:-4]}_inference_new.png', dpi=600)
#                 plt.close()
#         else:
#             pass
#
#         # print(glob.glob(file))
#     list_tiles = glob.glob('W:/home/shared/Machine_Learning_Datasets/Inference/*inference/')
# for grid in all_grids:
#     df = pd.read_csv(grid)
#     # df = apply_regex_to_df(df, r'\\', '/')
#     # df = apply_regex_to_df(df, 'Nasr/', 'W:/home/shared/Nasr_talos_2021/Nasr/')
#     df = apply_regex_to_df(df, 'W:/home/shared/Nasr_talos_2021/W:/home/shared/Nasr_talos_2021/Nasr/', 'W:/home/shared/Nasr_talos_2021/Nasr/')
#
#     df.to_csv(grid)
# # def worker(grid):



