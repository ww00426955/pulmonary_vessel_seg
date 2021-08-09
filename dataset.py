import os
import random
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset, DataLoader


class My_dataset(Dataset):
    def __init__(self, ct_root_dir, data_augment=False):
        self.data_augment = data_augment
        self.ct_root_dir = ct_root_dir
        self.ct_list = []
        for patient in os.listdir(self.ct_root_dir):
            patient_path = os.path.join(self.ct_root_dir, patient)
            for sub in os.listdir(patient_path):
                self.ct_list.append(os.path.join(patient_path, sub))



    def __getitem__(self, item):
        self.ct_path = self.ct_list[item]
        self.seg_path = self.ct_list[item].replace('ct_patch', 'label_patch')

        ct_array = sitk.GetArrayFromImage(sitk.ReadImage(self.ct_path))
        ct_seg_array = sitk.GetArrayFromImage(sitk.ReadImage(self.seg_path))
        # ct_seg_array[ct_seg_array > 0] = 1
        # if self.data_augment:
        #     ct_array, ct_seg_array = data_augment(ct_array, ct_seg_array)
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        ct_seg_array = torch.FloatTensor(ct_seg_array).unsqueeze(0)
        # if ct_seg_array.shape != (96, 128, 128):
        #     print(self.ct_path, ct_array.shape, ct_seg_array.shape)
        return ct_array, ct_seg_array

    def __len__(self):
        return len(self.ct_list)

if __name__ == '__main__':
    ct_root_path = r"/Files/Image_segment/datasets/blood_vessel/plan3_artery_period_zscore/val/ct_patch"
    my_dataset = My_dataset(ct_root_path)
    data_loader = DataLoader(my_dataset, 1)
    for index, (ct, seg) in enumerate(data_loader):
        pass
        # print(index, ct.shape, seg.shape)

