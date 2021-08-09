import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def read_dicom(path):
	reader = sitk.ImageSeriesReader()
	dicom_names = reader.GetGDCMSeriesFileNames(path)
	reader.SetFileNames(dicom_names)
	raw = reader.Execute()
	return raw

def visualization_CT(ct, mask, img_save_path):
        ct_flatten = ct.flatten()
        mask_flatten = mask.flatten()
        ct_res = ct_flatten[np.where(mask_flatten > 0)[0]]
        plt.hist(ct_res, bins=400)
        plt.savefig(img_save_path)
        plt.close()


if __name__ == '__main__':
    root_path = r"/Files/Xiaozhuang.Wang/PulmonaryData_100_425"
    save_path = r"/Files/Dongdong.Zhao/code/lung_vessel/data"
    low = -500
    high = 1000
    for sub_patient in tqdm.tqdm(os.listdir(root_path)):
        if os.path.exists(os.path.join(os.path.join(root_path, sub_patient, "dicom.nii.gz"))):
            ct_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub_patient, "dicom.nii.gz")))
            # Pulmonary vessels path
            PulmonaryArtery_path = os.path.join(root_path, sub_patient, "PulmonaryArtery.nii.gz")
            PulmonaryVein_path = os.path.join(root_path, sub_patient, "PulmonaryVeins.nii.gz")
		
            if os.path.exists(PulmonaryArtery_path):
                PulmonaryArtery = sitk.GetArrayFromImage(sitk.ReadImage(PulmonaryArtery_path))
                visualization_CT(ct_array, PulmonaryArtery, os.path.join(save_path, "PulmonaryArtery", sub_patient + ".png"))

            if os.path.exists(PulmonaryVein_path):

                PulmonaryVein = sitk.GetArrayFromImage(sitk.ReadImage(PulmonaryVein_path))
                visualization_CT(ct_array, PulmonaryVein, os.path.join(save_path,"PulmonaryVein", sub_patient + ".png"))
				
            if os.path.exists(PulmonaryArtery_path) and os.path.exists(PulmonaryVein_path):
			

                PulmonaryVessels = PulmonaryArtery + PulmonaryVein
                PulmonaryVessels[PulmonaryVessels > 1] = 1
                visualization_CT(ct_array, PulmonaryVessels, os.path.join(save_path, "PulmonaryVessels", sub_patient + ".png"))
		

