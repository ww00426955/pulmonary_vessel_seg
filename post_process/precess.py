import SimpleITK as sitk
import numpy as np

import sys
sys.path.append('..')

import skimage.morphology as morphology

from utils.utils import detect_endpoint, get_boundaries_from_mask, remove_small_volume
import imcut.pycut as pspc
import os




if __name__ == '__main__':

    root_path = r"C:\dongdong.zhao\project\Lung_vessel\val_raw_data"
    for sub in os.listdir(root_path):
        if os.path.exists(os.path.join(root_path, sub, "lung_5_blocks.nii.gz")):
            ct_path = os.path.join(root_path, sub, "dicom.nii.gz")
            lung_path = os.path.join(root_path, sub, "lung_5_blocks.nii.gz")

            data = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(lung_path))
            lung_mask[lung_mask > 0] = 1
            lung_mask = morphology.erosion(lung_mask)

            ROI = data * lung_mask

            bbox = get_boundaries_from_mask(ROI)
            ROI[ROI == 0] = -1000


            seeds_fg = np.zeros_like(data)
            seeds_bg = np.zeros_like(data)

            seeds_fg[ROI > 200] = 1
            seeds_bg[ROI < -300] = 2
            seeds = seeds_bg + seeds_fg


            #
            ROI = ROI[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
            seeds = seeds[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
            igc = pspc.ImageGraphCut(ROI, voxelsize=[1, 1, 1])
            igc.set_seeds(seeds)
            igc.run()
            res = np.zeros_like(data)
            res_roi = 1 - igc.segmentation
            res[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = res_roi

            res1 = remove_small_volume(res)

            sitk.WriteImage(sitk.GetImageFromArray(res1), os.path.join(root_path, sub, "ROI_predict.nii.gz"))
            print(sub, " done!")




