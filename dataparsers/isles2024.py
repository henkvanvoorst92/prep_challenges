import os
import pandas as pd
import numpy as np
def default_img_file_dict():
    return {  # use cropped cta files for further analyses
            'cbf': 'space-ncct_cbf.nii.gz',
            'cbv': 'space-ncct_cbv.nii.gz',
            'mtt': 'space-ncct_mtt.nii.gz',
            'tmax': 'space-ncct_tmax.nii.gz',
            # original dcm2niix and totalsegmentator
            'cta': 'space-ncct_cta.nii.gz',
            'ctp': 'space-ncct_ctp.nii.gz',
            'ncct':'_ncct.nii.gz',
            # cropping not required for MRAs -->use n4bfc
            'dwi_seg': '_lesion-msk.nii.gz',
            'adc': '_adc.nii.gz',
            'dwi': '_dwi.nii.gz',
            }

def default_database_file_dict():
    return {
        'outcome':'_outcome.csv',
        'baseline':'_baseline.csv'
    }

def all_files_df(path,
                 file_dcts=[]):
    """
    :param path: path with subdirectories containing files
                 all files will be described in the output dataframe
    :param file_dcts: list of dictionaries with key=name, value=part of filename
                        used to map all retrieve all file types
    :return: pd dataframe with all files and their subdirs
    """
    out = []
    for r,dr,files in os.walk(path):
        if len(files)<1:
            continue
        #ID = [d for d in r.split(os.sep) if 'sub-stroke' in d][0]
        for f in files:
            subs = f.split('_')
            ID = subs[0]
            if not 'sub-stroke' in ID:
                continue
            ses = subs[1]
            datatype = subs[-1]

            row = [ID, ses, datatype, r, dr, f, os.path.join(r,f)]
            out.append(row)
    out = pd.DataFrame(out)
    out.columns = ['ID', 'session', 'datatype', 'path', 'dir', 'f', 'file']
    out.index = out['ID']

    if len(file_dcts)>0:
        for dctname, dct in file_dcts.items():
            tmp = []
            for f in out['f']:
                img_type = [k for k, v in dct.items() if v in f]
                if len(img_type) == 1:
                    tmp.append(img_type[0])
                elif len(img_type) == 0:
                    tmp.append(np.NaN)
                else:
                    tmp.append(img_type.join(';'))
            out[dctname] = tmp

    return out

def clinical_database(data, col='clinical', p_sav=None):
    data = data[~data[col].isna()]

    out = []
    for ID, row in data.iterrows():
        tmp = pd.read_csv(row['file'])
        tmp = pd.concat([tmp.reset_index(drop=True),pd.DataFrame(row).T.reset_index(drop=True)], axis=1)
        out.append(tmp)
    out = pd.concat(out)
    out.index = out['ID']
    bl, outcome = out[out['clinical']=='baseline'], out[out['clinical']=='outcome']
    out = pd.concat([bl, outcome], axis=1)
    if p_sav is not None:
        out.to_excel(os.path.join(p_sav,'ISLES_baseline_outcome.xlsx'))
    return out

# Ensure the main function is executed only when the script is run directly
if __name__ == "__main__":
    #args = init_args()

    p = '/media/hvv/ec2480e5-6c18-468c-b971-5271432b386d/hvv/BL_NCCT/ISLES2024/ISLES24'
    isles_database_file_dict = default_database_file_dict()
    isles_img_file_dict = default_img_file_dict()

    #create dataframe with information of all files (including specific naming
    data = all_files_df(p, {'images': isles_img_file_dict, 'clinical':isles_database_file_dict})
    image_data = data[~data['images'].isna()]
    image_data.to_excel(os.path.join(p,'image_data.xlsx'))
    #construct a single file
    clinical_data = clinical_database(data, col='clinical', p_sav=p)
    #construct file with baseline and outcome data

    #

    print(1)


