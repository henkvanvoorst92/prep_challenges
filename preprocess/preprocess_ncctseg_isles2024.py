import os
import pandas as pd
import sys
import numpy as np
import shutil
import copy
from totalsegmentator.python_api import totalsegmentator
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from load import load_scans_from_dict, get_image_dict
sys.path.append('/home/hvv/Documents/git_repo')
from utils.registration import ants_register
from utils.utils import np2sitk
from utils.maskprocess import np_slicewise

if __name__ == "__main__":

    p = '/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/ISLES2024'
    data = pd.read_excel(os.path.join(p,'ISLES24', 'image_data.xlsx'))
    data.index = data['ID']
    #NCCT FLIP and register
    #segment brain in NCCT
    nnunet_dct = {'ncct':'_0000.nii.gz',
                  'ncct_flip':'_0001.nii.gz',
                  'ncct_ratio':'_0002.nii.gz',
                  'ctp':'ctp',
                  'roimask':'roimask',
                  'lesion':'lblTr'}
    idct = get_image_dict(data, nnunet_dct)

    lesion_folder = os.path.join(p,'lblTr')
    ctp_folder = os.path.join(p,'ctp')
    imagesTr = os.path.join(p,'imagesTr')
    roimask_folder = os.path.join(p,'roimask')

    os.makedirs(lesion_folder, exist_ok=True)
    os.makedirs(ctp_folder,exist_ok=True)
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(roimask_folder, exist_ok=True)

    rp = {'type_of_transform': 'TRSAA',
          'fix_bm': None,
          'mv_bm': None,
          'metric': 'mattes',
          'mask_all_stages': False,
          'default_value': -1024,
          'interpolator': 'linear'}

    for ID, d in tqdm(idct.items()):
        print('Running:', ID)
        filename_base = copy.copy(os.path.basename(d['ncct']))
        p_flipreg = os.path.join(imagesTr,copy.copy(filename_base).replace('.nii.gz',nnunet_dct['ncct_flip']))
        p_ratio = os.path.join(imagesTr,copy.copy(filename_base).replace('.nii.gz',nnunet_dct['ncct_ratio']))
        p_roimask = os.path.join(roimask_folder, f'{ID}_roimask.nii.gz')

        if not os.path.exists(p_roimask):
            totalsegmentator(d['ncct'], p_roimask,
                             ml=True, fast=False,
                             roi_subset=['brain', 'skull'], device='gpu',
                             verbose=False, nr_thr_saving=6, nr_thr_resamp=6)

        if not (os.path.exists(p_flipreg) and os.path.exists(p_ratio)):
            ncct = sitk.Cast(sitk.ReadImage(d['ncct']), sitk.sitkInt16)

            roimask = sitk.ReadImage(p_roimask)
            hm = (sitk.GetArrayFromImage(roimask) > 0) * 1
            hm = np_slicewise(hm, [binary_fill_holes], repeats=1, dim=0)

            flip = np2sitk(np.flip(sitk.GetArrayFromImage(ncct), axis=2), ncct)
            print('Registration')
            flipreg = ants_register(ncct,
                                    flip,
                                    rp=rp,
                                    clip_range=(-50,100))
            flipreg = sitk.Cast(flipreg,sitk.sitkInt16)
            print('Ratio image')
            ratio = sitk.GetArrayFromImage(ncct)/sitk.GetArrayFromImage(flipreg)
            ratio[np.isinf(ratio)] = 0
            ratio *= hm
            ratio = np2sitk(ratio.astype(np.float32),ncct)
            print('Saving')
            sitk.WriteImage(flipreg, p_flipreg)
            sitk.WriteImage(ratio, p_ratio)

        #data to new folder
        shutil.copy2(d['lesion'], lesion_folder)
        shutil.copy2(d['ctp'], ctp_folder)
        shutil.copy2(d['ncct'], os.path.join(imagesTr,copy.copy(filename_base).replace('.nii.gz',nnunet_dct['ncct'])))


