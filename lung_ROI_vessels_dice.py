import SimpleITK as sitk
import os

dice_list = []
root_path = r"D:\ProjectZhao\Lung_vessel\val_raw_data"
for sub in os.listdir(root_path):
    try:
        lung_roi = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "lung_5_blocks.nii.gz")))
        lung_roi[lung_roi > 0] = 1

        mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "PulmonaryVessels.nii.gz")))


        predict = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "Vessels_predict.nii.gz")))

        mask_roi = mask * lung_roi

        predict_roi = predict * lung_roi

        mask_roi_itk = sitk.GetImageFromArray(mask_roi)
        sitk.WriteImage(mask_roi_itk, os.path.join(root_path, sub, "ROI_vessels.nii.gz"))

        predict_roi_itk = sitk.GetImageFromArray(predict_roi)
        sitk.WriteImage(predict_roi_itk, os.path.join(root_path, sub, "ROI_predict.nii.gz"))
        mask_roi[mask_roi > 0] = 1
        dice = 2 * (mask_roi * predict_roi).sum() / (mask_roi.sum() + predict_roi.sum() + 1e-8)
        dice_list.append(dice)
        print("{",sub, "}", ":", dice)
    except:
        pass
print(sum(dice_list) / len(dice_list))