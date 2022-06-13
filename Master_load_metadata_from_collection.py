import sys
import glob
import pandas as pd

import xml.etree.ElementTree as ET
# input1 = sys.argv[1]
# input = sys.argv[2]
# output = sys.argv[3]


# print(f'Metadata input = {input1}GridSquare*/FoilHoles/*.xml AND {input}Atlas/Tile*.xml')
# print(f'Metadata output = {output}')
# foilholes_list = glob.glob(f'{input1}GridSquare*/FoilHoles/*.xml', recursive=True)
# highmag_list = glob.glob(f'{input1}GridSquare*/Data/*.xml', recursive=True)
# gridsquare_list = glob.glob(f'{input1}GridSquare*/*.xml', recursive=False)
# atlas_list = glob.glob(f'{input}Atlas/Tile*.xml', recursive=False)
# print('Found', len(highmag_list), 'dose measurement files')
# print('Found', len(atlas_list), 'atlas tiles')

def strip_values_from_xml(xml):
    keys = []
    values = []
    for event, elem in ET.iterparse(xml):
        if 'NominalMagnification' in elem.tag:
            keys.append('NominalMagnification')
            values.append(elem.text)
        if 'Key' in elem.tag:
            if 'KeyValue' in elem.tag:
                pass
            # elif 'FindFoil' in elem.text:
            # print('hello')
            # pass
            else:
                keys.append(elem.text)
        elif 'Value' in elem.tag:
            if 'numericValue' in elem.tag:
                pass
            elif 'PixelValue' in elem.tag:
                pass
            else:
                values.append(elem.text)
        elif elem.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}A':
            keys.append('A')
            values.append(elem.text)
        elif elem.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}B':
            keys.append('B')
            values.append(elem.text)
        elif elem.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}X':
            keys.append('X')
            values.append(elem.text)
        elif elem.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}Y':
            keys.append('Y')
            values.append(elem.text)
        elif elem.tag == '{http://schemas.datacontract.org/2004/07/Fei.SharedObjects}Z':
            keys.append('Z')
            values.append(elem.text)
        else:
            pass
    return keys, values


def metadata_dataframe(metadata_list):
    columns, values = strip_values_from_xml(metadata_list[0])
    columns.append('filename')
    metadata = pd.DataFrame(columns=columns)
    for file in metadata_list:
        cols, values = strip_values_from_xml(file)
        cols.append('filename')
        if cols == columns:
            values.append(file)
            df_length = len(metadata)
            metadata.loc[df_length] = values
        else:
            pass
    return metadata

def make_csv(list, name, normalize_by_gridsquare=False):
    metadata = metadata_dataframe(list)
    if 'DoseOnCamera' in metadata:
        metadata['Ice thickness'] = (-870.8 * metadata['DoseOnCamera'].astype(float) + 649.1)
        
        metadata['GridSquare'] = metadata['filename'].str.findall('(GridSquare_[0-9]*)')
        metadata['Ice thickness'] = metadata['Ice thickness'] + abs(metadata['Ice thickness'].min())
    metadata.to_csv(f'{name}')

    return metadata

# if len(foilholes_list) >= 1:
#     foilDATA = make_csv(foilholes_list, f'{output}/_foils.csv')
# if len(highmag_list) >= 1:
#     holeDATA = make_csv(highmag_list, f'{output}/_holes.csv')
# if len(gridsquare_list) >= 1:
#     gridDATA = make_csv(gridsquare_list, f'{output}/_squares.csv')
# if len(atlas_list) >= 1:
#     atlasDATA = make_csv(atlas_list, f'{output}/_atlas.csv')