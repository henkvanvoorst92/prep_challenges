
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import shutil
from tqdm import tqdm
import sys

###scripts to obtain data from database structure
def load_scans_from_dict(dct, skip_keys=['p_scan']):
    #returns a dict with all scans paths defined as values in dct
    #skip_keys are not returned
    dct_out = {}
    for k,v in dct.items():
        if k in skip_keys:
            continue
        if os.path.isfile(v):
            dct_out[k] = sitk.ReadImage(v)

    return dct_out

def get_image_dict(data, nnunet_dct):
    select_image_data = data[np.isin(data['images'],list(nnunet_dct.keys()))]
    image_dct = {k:{} for k in select_image_data.index.unique()}
    for ID, row in select_image_data.iterrows():
        image_dct[ID].update({row['images']:row['file']})
    return image_dct