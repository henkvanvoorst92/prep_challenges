import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import yaml
from dataparsers.topcow2024 import np2sitk, create_path_dict, load_data_from_dict, roi_edges_update_data_dict
from preprocess.ppmask import hdbet, bm_totalsegmentator
from tqdm import tqdm
#from scipy.ndimage import binary_fill_holes
#from utils.maskprocess import np_slicewise

if __name__ == "__main__":
    #args = init_args()

    p = '/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/TopCoW2024_Data_Release'
    #create a dict with all the paths
    dct = create_path_dict(p)

    for ID,dct_ID in tqdm(dct.items()):
        d = load_data_from_dict(dct_ID,
                                category_incl=['images', 'bm'],
                                modality_incl=['mr', 'ct'],
                                pp_roi_edges=False)

        #segment the brain for mr using HD-BET
        p_img = dct_ID['mr']['images'][0]
        file = os.path.basename(p_img)
        p_bm_mr = os.path.join(p, 'brainmask', file.replace('_0000', '_brainmask'))
        if len(dct_ID['mr']['bm'])<1:
            print('MR brain seg:', ID)
            #hdbet(p_img, p_bm_mr)
            bm_totalsegmentator(p_img, p_bm_mr, roi_subset=['brain'], task='total_mr')

        #segment the brain for CT using totalsegmentator
        p_img = dct_ID['ct']['images'][0]
        file = os.path.basename(p_img)
        p_bm_ct = os.path.join(p, 'brainmask', file.replace('_0000', '_brainmask'))
        if len(dct_ID['ct']['bm']) < 1:
            print('CT brain seg:', ID)
            bm_totalsegmentator(p_img, p_bm_ct, roi_subset = ['brain'], task='total')

        #update above without writing of the nifti files, keep in nib and use accordingly
        #if no mr is segmented experiment with removing high values or adapting normalization schemes

        #compute the distances of every point in the mask and a 5mm region around it



        print(1)