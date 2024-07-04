import os
import pandas as pd
import sys
import numpy as np

import SimpleITK as sitk
import shutil
from tqdm import tqdm
from load import load_scans_from_dict, get_image_dict
sys.path.append('/home/hvv/Documents/git_repo')
from utils.registration import ants_register

if __name__ == "__main__":

    p = '/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/ISLES2024'
    data = pd.read_excel(os.path.join(p,'image_data.xlsx'))

    #Todo
    #NCCT FLIP and register
    #segment brain in NCCT
    nnunet_dct = {'ncct':'_0000.nii.gz',
                  'ncct_flip':'_0001.nii.gz',
                  'ncct_ratio':'_0002.nii.gz',
                  'ctp':'ctp',
                  'lesion':'lblTr'}
    idct = get_image_dict(data, nnunet_dct)

    lesion_folder = os.path.join(p,'lblTr')
    ctp_folder = os.path.join(p,'ctp')
    imagesTr = os.path.join(p,'imagesTr')

    # os.makedirs(lesion_folder, exist_ok=True)
    # os.makedirs(ctp_folder,exist_ok=True)
    # os.makedirs(imagesTr, exist_ok=True)
    #
    # for ID, d in idct.items():
    #     #move lesion to dew
    #     shutil.copy2(d['lesion'], lesion_folder)
    #     shutil.copy2(d['ctp'], ctp_folder)
    #     shutil.copy2(d['ncct'], os.path.join(imagesTr,os.path.basename(d['ncct'].replace('.nii.gz',nnunet_dct['ncct']))))
    #
    #     ncct = sitk.ReadImage(d['ncct'])


