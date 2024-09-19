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
from utils.registration import ants_register, ants_register_roi_atlas
from utils.utils import np2sitk
from utils.maskprocess import np_slicewise

#try to use the scanprocessing version

if __name__ == "__main__":

    p_atlas_image = '/home/hvv/Documents/CT_atlas/template_with_skull.nii.gz'
    p_hemisphere = '/home/hvv/Documents/CT_atlas/template_with_skull_hemispheremask.nii.gz'
    atl = sitk.ReadImage(p_atlas_image)
    hemisphere_atl_mask = sitk.ReadImage(p_hemisphere)

    p = '/media/hvv/ec2480e5-6c18-468c-b971-5271432b386d/hvv/BL_NCCT/ISLES2024'
    data = pd.read_excel(os.path.join(p,'ISLES24', 'image_data.xlsx'))
    data.index = data['ID']
    #NCCT FLIP and register
    #segment brain in NCCT
    nnunet_dct = {'ncct':'_0000.nii.gz',
                    'ncct_flip':'_0001.nii.gz',
                    'ncct_ratio':'_0002.nii.gz',
                    'ctp':'_ctp.nii.gz',
                    'dwi_seg':'_dwi_mask.nii.gz',
                    'cbv': '_cbv.nii.gz',
                    'cbf': '_cbf.nii.gz',
                    'dwi': '_dwi.nii.gz',
                    'roimask': '_roimask.nii.gz',
                    'hemisphere': '_hemispheremask.nii.gz'
                  }
    idct = get_image_dict(data, nnunet_dct)

    #lesion_folder = os.path.join(p,'lblTr')
    imagesTr = os.path.join(p,'NCCT-3chan')
    for k in nnunet_dct.keys():
        if 'ncct' not in k:
            os.makedirs(os.path.join(p,k), exist_ok=True)

    # os.makedirs(lesion_folder, exist_ok=True)
    # os.makedirs(ctp_folder,exist_ok=True)
    os.makedirs(imagesTr, exist_ok=True)
    # os.makedirs(roimask_folder, exist_ok=True)
    # os.makedirs(hemisphere_folder, exist_ok=True)

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
        p_roimask = os.path.join(p, 'roimask', f'{ID}'+nnunet_dct['roimask']+'.nii.gz')
        p_hemispherereg = os.path.join(p,'hemisphere', copy.copy(filename_base).replace('.nii.gz',nnunet_dct['hemisphere']))
        ncct = None

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
            sitk.WriteImage(ncct,
                            os.path.join(imagesTr, copy.copy(filename_base).replace('.nii.gz', nnunet_dct['ncct'])))

        if (not os.path.exists(p_hemispherereg)) and (os.path.exists(p_hemisphere)):
                #coregister halfbrain mask for laterality selection (using FU DWI or CTP core/penumbra)
                if ncct is None:
                    ncct = sitk.ReadImage(d['ncct'])
                #coregister the hemisphere atlas mask
                hemispheres = ants_register_roi_atlas(ncct,
                                                      atl,
                                                      hemisphere_atl_mask,
                                                      rp = rp,
                                                      clip_range=(0, 100))
                sitk.WriteImage(hemispheres, p_hemispherereg)
                #run whole script again after NCCT lesion segmentation is done
                #identification_mask_preference = [d['dwi_pierre'], d['core_mask'], d['dl_ncct_mask']]

        try:
            im = sitk.Cast(im, sitk.sitkInt16)
        except:
            im = im
        # data to new folder

        print('Copying other files')
        #ncct and roimask require further processing hence not use
        for k in nnunet_dct.keys():
            if not ('ncct' in k or 'roimask' in k or 'hemisphere' in k):
                f_out = os.path.join(p, k, ID+nnunet_dct[k])

                if k in ['cbv', 'cbf', 'tmax']:
                    im = sitk.ReadImage(d[k])
                    im = sitk.Cast(im, sitk.sitkFloat32)
                    sitk.WriteImage(im, f_out)

                if (not os.path.exists(f_out)) and os.path.exists(d[k]):
                    im = sitk.ReadImage(d[k])
                    sitk.WriteImage(im, f_out)

        print('Finished')



