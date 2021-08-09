import os
import SimpleITK as sitk
from utils.utils import visualization_CT, remove_small_volume
import numpy as np
from skimage.morphology import erosion


def constant_threshold_seg(data, threshold=-600):
    res = (data > threshold).astype(np.int8)
    res = remove_small_volume(res, threshold=10000)
    return res

root_path = r"C:\Users\dongdong.zhao\Desktop\sss\108"
lung = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, "lung_5_blocks.nii.gz")))
lung[lung > 0] = 1
lung = erosion(erosion(lung))


dicom = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, "dicom.nii.gz")))

ROI = dicom * lung
ROI[ROI == 0] = -1000

# visualization_CT(dicom, lung, r"C:\Users\dongdong.zhao\Desktop\res.png")
res1 = constant_threshold_seg(ROI)
sitk.WriteImage(sitk.GetImageFromArray(res1), r"C:\Users\dongdong.zhao\Desktop\sss\108\constant_threshold_seg.nii.gz")

