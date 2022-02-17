import pandas as pd
import mrcfile
import glob
import numpy as np
import cupy
import cupyx.scipy.ndimage as cpx
import statsmodels.api as sm
import cucim.skimage.morphology as cumorph
import matplotlib.pyplot as plt



def flip_and_rotate_gridsquares(glob): ## correct for distortions
    for x in glob:
        with mrcfile.open(x) as img:
            img = np.rot90(img.data, k=1)
            img = np.fliplr(img)
            np.save(f'corrected_{x}', img, allow_pickle=True)

def cupy_calc_size_distance(blob, thres=20, target_size=60, max_size=600):
    blob, list_sum = find_holes(blob, thres, target_size, max_size)
    blob, num_dots = cpx.label(blob)
    if num_dots > 1:
        list_index = cupy.arange(1, num_dots + 1)
        list_cent = cupy.asarray(cpx.measurements.center_of_mass(blob, labels=blob, index=list_index))
    else:
        return [], 1, blob
    return list_cent, num_dots, blob

def find_holes(array, thres, target_size, max_size):
    array = array * (255.0/array.max())
    array = cupy.asarray(array > thres, dtype=cupy.float32)
    array = cumorph.opening(array, cumorph.disk(1))
    lbls, nr_of_lbls = cpx.label(array)
    lbl = cupy.arange(1, nr_of_lbls + 1)
    list_of_sizes = []
    for l in lbl:
        subarray = lbls[lbls == l]
        size = subarray.size
        if size < target_size:
            lbls[lbls == l] = 0
        if size > max_size:
            lbls[lbls == l] = 0
        else:
            lbls[lbls == l] = 1
            list_of_sizes.append(size)
    return lbls, list_of_sizes

def calculate_holes(original, array, list_of_centers, dilation=64):
    array[array > 0] = 1
    masked_array = original * array
    def get_mean(tuple, masked_array=masked_array, dilation=dilation):
        y, x = tuple
        x_1 = int(y - dilation)
        x_2 = int(y + dilation)
        y_1 = int(x - dilation)
        y_2 = int(x + dilation)
        slice = masked_array[x_1:x_2, y_1:y_2]
        slice = slice[cupy.nonzero(slice)]
        slice = slice[slice > (cupy.mean(slice) - 750)]
        mean = cupy.mean(slice)
        return mean
    output = cupy.array(list(map(get_mean, list_of_centers)))
    return output

def get_dataframe(tile, name):
    list_cent, num_dots, blob = cupy_calc_size_distance(tile, thres=50)
    if num_dots == 1:
        column_names = ["a", "b", "c"]
        df = pd.DataFrame(columns=column_names)
        return df
    else:
        means_of_all_holes = calculate_holes(tile, blob, list_cent, dilation=3)
        means_of_all_holes = [i.get() for i in means_of_all_holes]

        data = np.c_[list_cent.get(), np.array(means_of_all_holes)]
        dataframe = pd.DataFrame(data, columns=['x', 'y', 'means'])
        dataframe.to_csv(f'{name[:-4]}_raw.csv')
        return dataframe

def regression_statsmodels(dataframe, name):
  rlm = sm.RLM(dataframe[[2]], sm.add_constant(dataframe[[4]]), M=sm.robust.norms.HuberT())
  rlm_results = rlm.fit()
  line_X = np.arange(np.min(dataframe[4]), np.max(dataframe[4]), 0.2).reshape(-1, 1)
  line_y_rlm = rlm_results.predict(sm.add_constant(line_X))
  y_pred = rlm_results.predict(sm.add_constant(dataframe[4]))
  dataframe.plot.scatter(x=4, y=2)
  u = np.mean(abs(rlm_results.resid)) * 100 / np.max(dataframe[2])
  plt.title(f'average error ∓ {u}%')
  plt.plot(line_X, line_y_rlm, c="black", label="statsmodels_regression")
  plt.legend(loc='lower right')
  plt.savefig(f'{name}_regression2.png')
  plt.close()
  return y_pred

def surface_fit(dataframe, dataframe_empty, example, name):
    mean_normal = []
    print(dataframe)
    for idx, row in dataframe.iterrows():
        x, y, mean = row
        blank = 0
        range = 100
        slice = dataframe_empty[dataframe_empty.x.values > (x - range)]
        slice = slice[slice.x.values < (x + range)]
        slice = slice[slice.y.values > (y - range)]
        slice = slice[slice.y.values < (y + range)]
        if slice.empty:
            range = 200
            slice = dataframe_empty[dataframe_empty.x.values > (x - range)]
            slice = slice[slice.x.values < (x + range)]
            slice = slice[slice.y.values > (y - range)]
            slice = slice[slice.y.values < (y + range)]
            if slice.empty:
                blank = dataframe_empty['means'].mean()
            else:
                blank = slice['means'].mean()
        else:
            blank = slice['means'].mean()

        mean = mean + (1000 - blank)
        mean_normal.append(mean)
    dataframe['corrected_mean'] = mean_normal
    dataframe.plot.scatter(x='y', y='x', c='corrected_mean', cmap='inferno', s=1, vmax=1000, vmin=500)
    plt.imshow(example.get(), cmap='gray')
    plt.savefig(f'{name[:-4]}after.png', dpi=800)
    plt.close()
    dataframe.to_csv(f'{name[:-4]}range_and_flat_corrected.csv')
    return dataframe

def measure_atlas(tiles):
    for tile in tiles:
        example = mrcfile.open(tile)
        example = np.rot90(example.data, k=1)
        example = np.fliplr(example)
        example = cupy.array(example)
        example = cupy.pad(example, 10, constant_values=0)
        dataframe = get_dataframe(example, tile)
        if dataframe.empty:
            print('bad tile')
        else:
            dataframe_empty = dataframe[dataframe.means.values > 5800]
            dataframe['means'] = dataframe['means'] / 6.3272
            dataframe_empty['means'] = dataframe_empty['means'] / 6.3272 ## parameter for scaling atlas mag to gridsquare
            dataframe.plot.scatter(x='y', y='x', c='means', cmap='inferno', s=1, vmax=1000, vmin=500)
            plt.imshow(example.get(), cmap='gray')
            plt.savefig(f'{tile[:-4]}before.png', dpi=800)
            plt.close()
            dataframe = surface_fit(dataframe, dataframe_empty, example, tile)
            dataframe['thickness'] = (dataframe['corrected_mean']*-0.3560) + 335.9092 ### parameters from linear regression on gridsquare mag
            dataframe.plot.scatter(x='y', y='x', c='thickness', cmap='inferno_r', s=1, vmax=250, vmin=0)
            plt.imshow(example.get(), cmap='gray')
            plt.savefig(f'{tile[:-4]}_thickness.png', dpi=800)
            plt.close()


def produce_histogram(frames):
    completed_analysis = glob.glob('*range_and_flat*.csv')
    for data in completed_analysis:
        dataframe = pd.read_csv(data)
        dataframe['thickness'] = (dataframe['corrected_mean'] * -0.3560) + 335.9092
        dataframe = dataframe[dataframe.thickness.values > 20]
        for idx, row in dataframe.iterrows():
            ix, x, y, mean, corrected_mean, thickness = row
            range = 40
            slice = dataframe[dataframe.x.values > (x - range)]
            slice = slice[slice.x.values < (x + range)]
            slice = slice[slice.y.values > (y - range)]
            slice = slice[slice.y.values < (y + range)]
            if slice.shape[0] < 3:
                dataframe['thickness'][idx] = 0
        dataframe.to_csv(f'{data}_outliers_removed.csv')

def combine_dataframes(example):
    data = glob.glob('*_outliers_removed.csv')
    master = pd.read_csv(data[0])
    master.plot.scatter(x='y', y='x', c='thickness', cmap='inferno_r', s=1, vmax=250, vmin=0)
    plt.imshow(example.get(), cmap='gray')
    plt.savefig(f'{data[0][:-8]}_thickness.png', dpi=800)
    plt.close()
    for dtF in data[1:]:
        x = pd.read_csv(dtF)
        master = master.append(x, ignore_index=True,)
    master.to_csv('Atlas_analysis.csv')
    return master

tiles = glob.glob('Tile*.mrc')
measure_atlas(tiles)
completed_analysis = glob.glob('*range_and_flat*.csv')
produce_histogram(completed_analysis)
images = glob.glob('Tile*.mrc')
example = mrcfile.open(images[0])
example = np.rot90(example.data, k=1)
example = np.fliplr(example)
example = cupy.array(example)
example = cupy.pad(example, 10, constant_values=0)
combine_dataframes(example)