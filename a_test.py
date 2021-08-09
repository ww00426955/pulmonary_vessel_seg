import os
import SimpleITK as sitk
import numpy as np
path1 = r"C:\Users\dongdong.zhao\Desktop\sss\108\PulmonaryVessels.nii"

data = sitk.GetArrayFromImage(sitk.ReadImage(path1))
print(data.shape)
mask_voxel_coords = data[np.where(data != 0)] * 4
print(mask_voxel_coords.max())