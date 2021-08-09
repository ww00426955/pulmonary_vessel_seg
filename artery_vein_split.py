"""
使用模型一分割肺血管的结果   与   模型二动静脉分类的结果做处理，得出最后结果

"""

import os
import SimpleITK as sitk
import numpy as np
import skimage.morphology as morphology
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from utils.utils import keep_maximum_volume

artery_list = []
vein_list = []
root_path = r"C:\Users\dongdong.zhao\Desktop\sss"
for sub in os.listdir(root_path):
    if os.path.exists(os.path.join(root_path, sub, "lung_5_blocks.nii.gz")):
        ROI_vessels_path = os.path.join(root_path, sub, "constant_threshold_seg.nii.gz")
        vessels_predict_path = os.path.join(root_path, sub, "Vessels_predict.nii.gz")
        vein_predict_path = os.path.join(root_path, sub, "Vessels_res2.nii.gz")
        artery_predict_path = os.path.join(root_path, sub, "Vessels_res1.nii.gz")

        gt_vein = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "PulmonaryVeins.nii.gz")))
        gt_artery = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "PulmonaryArtery.nii.gz")))

        vessels_predict = sitk.GetArrayFromImage(sitk.ReadImage(vessels_predict_path))
        ROI_vessels = sitk.GetArrayFromImage(sitk.ReadImage(ROI_vessels_path))
        lung = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "lung_5_blocks.nii.gz")))
        lung[lung > 0] = 1
        # vessels_predict = ROI_vessels + vessels_predict * (1 - lung)
        vessels_predict[vessels_predict > 0] = 1
        sitk.WriteImage(sitk.GetImageFromArray(vessels_predict),os.path.join(root_path, sub, "tmp.nii.gz"))

        threashold = 0.3   # 阈值越低，细节越多
        # 静脉
        vein_predict_probability = sitk.GetArrayFromImage(sitk.ReadImage(vein_predict_path))
        vein_predict = (vein_predict_probability > threashold).astype(np.int8)
        vein_predict = morphology.dilation(vein_predict)

        artery_tmp = keep_maximum_volume(vessels_predict - vein_predict)
        artery_tmp = np.clip(artery_tmp, 0, 1)
        vein_res = keep_maximum_volume(vessels_predict - artery_tmp)

        # 动脉
        artery_predict_probability = sitk.GetArrayFromImage(sitk.ReadImage(artery_predict_path))
        artery_predict = (artery_predict_probability > threashold).astype(np.int8)
        artery_predict = morphology.dilation(artery_predict)
        vein_tmp = keep_maximum_volume(vessels_predict - artery_predict)
        vein_tmp = np.clip(vein_tmp, 0, 1)
        artery_res = keep_maximum_volume(vessels_predict - vein_tmp)

        splited_vessels = artery_res + vein_res * 2

        # 处理不确定类别的血管 要不然对各个不确定部位进行像素概率和统计，如果该连通域中动脉和大于静脉和，则判定为动脉
        unsure_mask = (splited_vessels == 3).astype(np.int8)
        label_mask, num = label(unsure_mask, return_num=True)
        # print(num)  # 16
        properties = regionprops(label_mask)
        for sub_property in properties:

            valid_label = set()
            valid_label.add(sub_property.label)
            sub_res = np.in1d(label_mask, list(valid_label)).reshape(label_mask.shape).astype(np.int8)
            # 标签中先对这一部分清零
            splited_vessels[sub_res == 1] = 0
            vein_sub_probability = np.sum(vein_predict_probability * sub_res)
            artery_sub_probability = np.sum(artery_predict_probability * sub_res)
            if vein_sub_probability > artery_sub_probability:
                sub_res *= 2
            splited_vessels += sub_res


        # # 肺内部dice
        # lung = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root_path, sub, "lung_5_blocks.nii.gz")))
        # lung[lung > 0] = 1
        # artery_res = artery_res * lung
        # gt_artery = gt_artery * lung
        # vein_res = vein_res * lung
        # gt_vein = gt_vein * lung

        dice_artery = 2 * (artery_res * gt_artery).sum() / (artery_res.sum() + gt_artery.sum() + 1e-8)
        dice_vein = 2 * (vein_res * gt_vein).sum() / (vein_res.sum() + gt_vein.sum() + 1e-8)
        artery_list.append(dice_artery)
        vein_list.append(dice_vein)
        print("artery:{}, vein:{}".format(dice_artery, dice_vein))

        sitk.WriteImage(sitk.GetImageFromArray(splited_vessels),
                        os.path.join(root_path, sub, "Vessels_predict_classification.nii.gz"))

# patient_list = [102, 108, 115, 116, 117, 118, 138, 148, 149, 154, 156, 17, 189, 228, 237, 244, 248, 253]
#
# artery_result = round(sum(artery_list) / len(artery_list), 3)
# vein_result = round(sum(vein_list) / len(vein_list), 3)
#
# bar_width = 0.2
# index1 = np.arange(len(patient_list))
# index2 = index1 + bar_width * 1
# index3 = index1 + bar_width * 2
#
# model = ("artery=" + str(artery_result),
#          "vein=" + str(vein_result))
#
# plt.figure(figsize=(16, 9))
#
# plt.bar(index1, height=artery_list, width=bar_width, color='aqua', label=model[0])
# plt.bar(index2, height=vein_list, width=bar_width, color='fuchsia', label=model[1])
#
# # 均值
# plt.hlines(artery_result, 0, len(patient_list), colors='aqua', linestyles='--')
# plt.hlines(vein_result, 0, len(patient_list), colors='fuchsia', linestyles='--')
#
# plt.ylim((0.5, 0.9))
# plt.legend(loc='upper right')
# plt.xticks(index1 + bar_width / 2, patient_list)
# plt.ylabel("DICE")
# plt.title("测试数据的dice评价指标")
# plt.show()




