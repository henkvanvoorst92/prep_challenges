import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import yaml


def np2sitk(arr: np.ndarray, original_img: sitk.SimpleITK.Image):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    #img.CopyInformation(original_img)
    return img


# Function to retrieve files by type (ct or mr) for a given folder and ID
def get_files_by_type(folder, id_number, file_type):
    return [os.path.join(folder,f) for f in os.listdir(folder) if f"_{file_type}_{id_number}" in f]

def create_path_dict(p):
    p_img = os.path.join(p, 'imagesTr')
    p_seg = os.path.join(p, 'cow_seg_labelsTr')
    p_edges = os.path.join(p, 'antpos_edges_labelsTr')
    p_roi = os.path.join(p, 'roi_loc_labelsTr')
    p_bm = os.path.join(p,'brainmask')

    # get all IDs
    IDs = [f.replace('.nii.gz', '').split('_')[-1] for f in os.listdir(p_seg)]

    data_dict = {}
    for id_number in IDs:
        data_dict[id_number] = {
            'ct': {
                'images': get_files_by_type(p_img, id_number, 'ct'),
                'seg': get_files_by_type(p_seg, id_number, 'ct'),
                'edges': get_files_by_type(p_edges, id_number, 'ct'),
                'roi': get_files_by_type(p_roi, id_number, 'ct'),
                'bm': get_files_by_type(p_bm, id_number, 'ct'),
            },
            'mr': {
                'images': get_files_by_type(p_img, id_number, 'mr'),
                'seg': get_files_by_type(p_seg, id_number, 'mr'),
                'edges': get_files_by_type(p_edges, id_number, 'mr'),
                'roi': get_files_by_type(p_roi, id_number, 'mr'),
                'bm': get_files_by_type(p_bm, id_number, 'mr'),
            }
        }

    return data_dict



def load_data_from_dict(dct,
                        category_incl=['images', 'seg', 'edges', 'roi', 'bm'],
                        modality_incl=['mr', 'ct'],
                        pp_roi_edges=False,
                        skip_keys=['LICENSE.txt', 'README.txt']):
    # Returns a dict with all scans and other files as defined in the nested structure of dct
    # Skip keys are not processed
    #dict should represent data from a single ID (not all!).


    dct_out = {}
    for modality, categories in dct.items():
        if modality not in modality_incl:
            continue

        dct_out[modality] = {}
        for category, file_list in categories.items():
            if category not in category_incl:
                continue

            dct_out[modality][category] = []
            for file_path in file_list:
                if file_path in skip_keys:
                    continue
                if os.path.isfile(file_path):
                    if file_path.endswith(('.nii', '.nii.gz')):
                        # Read NIfTI files
                        dct_out[modality][category].append(sitk.ReadImage(file_path))
                    elif file_path.endswith('.yml'):
                        # Read YAML files
                        with open(file_path, 'r') as f:
                            yaml_data = yaml.safe_load(f)
                            dct_out[modality][category].append(yaml_data)
                    elif file_path.endswith('.txt'):
                        # Read text files
                        with open(file_path, 'r') as f:
                            txt_data = f.read()
                            dct_out[modality][category].append(txt_data)
        if pp_roi_edges:
            dct_out = roi_edges_update_data_dict(dct_out)

    return dct_out


def process_roi_data(roi_data):
    """
    Extracts size and location from ROI data and returns them as NumPy arrays.

    Args:
        roi_data (str): A string containing ROI information.

    Returns:
        tuple: A tuple containing two NumPy arrays (size, location).
    """
    size_line = roi_data.split('\n')[1]
    location_line = roi_data.split('\n')[2]

    # Extract the numbers from the strings
    size = np.array(list(map(int, size_line.split(':')[1].strip().split())))
    location = np.array(list(map(int, location_line.split(':')[1].strip().split())))

    return size, location

def process_edges_data(edges_data, id=None):
    """
    Processes the edges data to create one-hot encoded tables for anterior and posterior.

    Args:
        edges_data (list): A list of dictionaries containing edge data.
        id (str, optional): An optional ID to include in the DataFrame.

    Returns:
        tuple: A tuple containing two Pandas DataFrames (anterior_df, posterior_df).
    """
    anterior_data = edges_data.get('anterior', {})
    posterior_data = edges_data.get('posterior', {})

    # One-hot encode the anterior and posterior data
    anterior_df = pd.DataFrame(anterior_data, index=[0])
    posterior_df = pd.DataFrame(posterior_data, index=[0])

    # If ID is provided, add it as a column
    if id is not None:
        anterior_df['ID'] = id
        posterior_df['ID'] = id

    # Ensure consistent column order
    anterior_df = anterior_df.reindex(sorted(anterior_df.columns), axis=1)
    posterior_df = posterior_df.reindex(sorted(posterior_df.columns), axis=1)
    return anterior_df, posterior_df


def roi_edges_update_data_dict(dct):
    """
    Processes ROI and edges data from the dictionary for a given ID and updates the dictionary with processed data.

    Args:
        dct_ID (dict): Dictionary for a single ID containing modalities and categories.
        id (str, optional): ID of the data, used for tagging in DataFrames.

    Returns:
        dict: Updated dictionary with processed ROI and edges data.
    """
    # Load data from dictionary

    dct = dct.copy()

    for modality, categories in dct.items():
        # Process ROI data
        if 'roi' in categories:
            roi_text = categories['roi']
            if roi_text:
                size, location = process_roi_data(roi_text[0])
                dct[modality]['roi_size'] = size
                dct[modality]['roi_location'] = location

                #create mask if imaging data available
                if 'images' in categories:
                    img = categories['images'][0]
                    img_np = sitk.GetArrayFromImage(img)
                    mask = np.zeros_like(img_np)
                    start_indices = location
                    end_indices = start_indices + size + 1
                    #check if this should be inverted or not
                    mask[start_indices[2]:end_indices[2],
                                start_indices[1]:end_indices[1],
                                start_indices[0]:end_indices[0]] = 1

                    # mask[start_indices[0]:end_indices[0],
                    #             start_indices[1]:end_indices[1],
                    #             start_indices[2]:end_indices[2]] = 1

                    dct[modality]['roi_mask_np'] = mask.astype(np.int16)
                    dct[modality]['roi_mask_nii'] = sitk.Cast(np2sitk(mask, img), sitk.sitkInt16)

        # Process edges data
        if 'edges' in categories:
            edges_yaml = categories['edges']
            if edges_yaml:
                anterior_df, posterior_df = process_edges_data(edges_yaml[0], id=None)
                dct[modality]['edges_df'] = pd.concat([anterior_df, posterior_df], axis=1)
    return dct


# Ensure the main function is executed only when the script is run directly
if __name__ == "__main__":
    #args = init_args()

    p = '/media/hvv/71672b1c-e082-495c-b560-a2dfc7d5de59/data/TopCoW2024_Data_Release'
    #create a dict with all the paths
    dct = create_path_dict(p)

    for ID,dct_ID in dct.items():
        d = load_data_from_dict(dct_ID, pp_roi_edges=True)
        #d2 = roi_edges_update_data_dict(d)


        #


        #comput the distance map for both mr and ct and store it, consider also storing the bounding box

        #sitk.WriteImage(d2['ct']['roi_mask_nii'], os.path.join(p,ID+'_mask.nii.gz'))

        break






    print(1)


