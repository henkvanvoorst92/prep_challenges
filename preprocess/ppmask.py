from totalsegmentator.python_api import totalsegmentator
import os
import nibabel as nib
def hdbet(p_in, p_out, overwrite=False):
    #-i and -o can be folder or filename
    #-s: 1 if you want to save the brainmask
    #-b: 1 if you want to save the processed mri
    #-device: number is gpu number otherwise cpu (=slow)
    #-mode: accurate or fast
    ovrwr = 1 if overwrite else 0
    cmd = f'hd-bet -i {p_in} -o {p_out} -s 1 -b 0 -device 0 -mode accurate --overwrite_existing {ovrwr}'
    print(cmd)
    os.system(cmd)

def bm_totalsegmentator(p_in, p_out, roi_subset=['brain', 'skull'], task='total'):
    if not os.path.exists(p_out):
        totalsegmentator(p_in, p_out, #.replace('.nii','').replace('.gz','')
                         ml=True, fast=False,
                         roi_subset=roi_subset, device='gpu', task=task,
                         verbose=False, nr_thr_saving=6, nr_thr_resamp=6)

def nowrite_bm_totalsegmentator(p_in, p_out, roi_subset=['brain', 'skull'], task='total'):
    if not os.path.exists(p_out):
        img = nib.loab
        totalsegmentator(p_in, p_out, #.replace('.nii','').replace('.gz','')
                         ml=True, fast=False,
                         roi_subset=roi_subset, device='gpu', task=task,
                         verbose=False, nr_thr_saving=6, nr_thr_resamp=6)
