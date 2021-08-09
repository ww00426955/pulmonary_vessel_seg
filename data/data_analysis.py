import numpy as np
import SimpleITK as sitk
import os

def read_dicom(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    raw = reader.Execute()
    return raw

root_path = r"/Files/Xiaozhuang.Wang/PulmonaryData_100_425"


ct_value = []
space0 = []
space1 = []
space2 = []

for i, sub_patient in enumerate(os.listdir(root_path)):
    try:
        ct_itk = sitk.ReadImage(os.path.join(root_path, sub_patient, "dicom.nii.gz"))
        # spacing information
        spacing = ct_itk.GetSpacing()
        space0.append(spacing[0])
        space1.append(spacing[1])
        space2.append(spacing[2])
        space0.sort()
        space1.sort()
        space2.sort()


        ct_array = sitk.GetArrayFromImage(ct_itk)
        PulmonaryArtery = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub_patient, "PulmonaryArtery.nii.gz")))
        PulmonaryVein = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub_patient, "PulmonaryVeins.nii.gz")))
        total_vessel = PulmonaryArtery + PulmonaryVein

        total_vessel[total_vessel > 1] = 1

        ct_flatten = ct_array.flatten()
        vessel_flatten = total_vessel.flatten()
        masked_ct = ct_flatten[np.where(vessel_flatten > 0)[0]]
        ct_value.extend(masked_ct)
        ct_value.sort()

        if i % 10 == 0:
            print("ct clip", ct_value[int(len(ct_value) * 0.05)], ct_value[int(len(ct_value) * 0.98)])
            print("ct mean", np.mean(ct_value))
            print("ct std", np.std(ct_value))
            print("median spacing: ", space0[int(len(space0) / 2)], ", ", space1[int(len(space1) / 2)], ", ", space2[int(len(space2) / 2)])

    except:
        print(sub_patient, ": data is in incomplete!")

