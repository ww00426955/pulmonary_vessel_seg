import time
import os
import torch
from cfg import cfg
import model_zoo
import numpy as np
import SimpleITK as sitk
from utils.utils import keep_maximum_volume, overlap_predict, get_boundaries_from_mask
import threading
from torch.nn import functional as F
from multiprocessing import cpu_count, Pool
from data_preprocess import PreProcess
import itertools
from functools import partial
import math


def overlap_predict2(net, ct, patch_size, rate=2):
    """
    :param net: 网络模型
    :param data_for_predict:   输入的待预测矩阵
    :param patch_size:     切块大小
    :param rate=2   1/2重叠滑块
    :return: 预测输出结果   与输入大小一致
    """

    # slide_rate = patch_size // rate  # 滑块的步长，默认为patch_size的一半（64）
    slide_rate = list(map(lambda x: x // rate, patch_size))
    pad_num_z = (patch_size[0] - ct.shape[0] % patch_size[0]) if (
            ct.shape[0] <= patch_size[0]) else (slide_rate[0] - ct.shape[0] % slide_rate[0])
    pad_num_y = (patch_size[1] - ct.shape[1] % patch_size[1]) if (
            ct.shape[1] <= patch_size[1]) else (slide_rate[1] - ct.shape[1] % slide_rate[1])
    pad_num_x = (patch_size[2] - ct.shape[2] % patch_size[2]) if (
            ct.shape[2] <= patch_size[2]) else (slide_rate[2] - ct.shape[2] % slide_rate[2])

    tmp_ct = np.pad(ct, ((0, pad_num_z), (0, pad_num_y), (0, pad_num_x)), 'constant')


    z_slide_num = math.ceil((tmp_ct.shape[0] - patch_size[0]) / slide_rate[0]) + 1
    y_slide_num = math.ceil((tmp_ct.shape[1] - patch_size[1]) / slide_rate[1]) + 1
    x_slide_num = math.ceil((tmp_ct.shape[2] - patch_size[2]) / slide_rate[2]) + 1
    # 保存最终的预测结果
    tmp_res = np.zeros(tmp_ct.shape)
    tmp_res1 = np.zeros(tmp_ct.shape)
    # 用于计算滑块过程中每个像素重叠的次数
    num_array = np.zeros(tmp_res.shape)

    with torch.no_grad():
        for xx in range(x_slide_num):
            for yy in range(y_slide_num):
                for zz in range(z_slide_num):
                    ct_part = tmp_ct[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                              yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                              xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]]


                    # 当下滑块对应的像素点滑块次数+1
                    num_array[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += 1

                    ct_tensor = torch.FloatTensor(ct_part).cuda()
                    ct_tensor = ct_tensor.unsqueeze(dim=0)
                    ct_tensor = ct_tensor.unsqueeze(dim=0)


                    outputs, outputs1 = net(ct_tensor)
                    outputs = outputs.squeeze().cpu().detach().numpy()
                    outputs1 = outputs1.squeeze().cpu().detach().numpy()
                    # 将预测的结果加入到对应的位置上
                    tmp_res[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += outputs
                    tmp_res1[zz * slide_rate[0]:zz * slide_rate[0] + patch_size[0],
                    yy * slide_rate[1]:yy * slide_rate[1] + patch_size[1],
                    xx * slide_rate[2]:xx * slide_rate[2] + patch_size[2]] += outputs1
    tmp_res = tmp_res / num_array  # 对应像素点叠加预测的结果除以相对应的次数，得到预测平均值
    tmp_res1 = tmp_res1 / num_array
    return tmp_res[0:ct.shape[0], 0:ct.shape[1], 0:ct.shape[2]], tmp_res1[0:ct.shape[0], 0:ct.shape[1], 0:ct.shape[2]]

def get_path(test_root_path):
    result = []
    for sub in os.listdir(test_root_path):
        result.append(os.path.join(test_root_path, sub))
    return result

def predict(cfg, net, patient_path):
    if os.path.exists(os.path.join(patient_path, "lung_5_blocks.nii.gz")):
        p = PreProcess(cfg)
        processed_ct, origial_size = p.read_and_normalization_CT(patient_path)
        patient_name = patient_path.split('\\')[-1]


        # 肺部ROImask
        lung_mask = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(patient_path, "lung_5_blocks.nii.gz")))
        # lung_mask = torch.Tensor(lung_mask).unsqueeze(0).unsqueeze(0)
        # lung_mask = F.interpolate(lung_mask, size=processed_ct.shape, mode='nearest')
        # lung_mask = lung_mask.squeeze().cpu().detach().numpy()
        bbox = get_boundaries_from_mask(lung_mask)
        ROI = processed_ct[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]

        # 肺部区域重采样到[160， 160， 256]
        ROI_size = ROI.shape
        ROI = torch.Tensor(ROI).unsqueeze(0).unsqueeze(0).to(cfg.device)
        ROI = torch.nn.functional.interpolate(ROI, size=[160, 160, 256], mode='trilinear')
        with torch.no_grad():
            ROI_res = net(ROI)
            ROI_res = torch.nn.Softmax(dim=1)(ROI_res)



        # ROI_res = ROI_res.squeeze().cpu().numpy()
        # sitk.WriteImage(sitk.GetImageFromArray(ROI_res[0]), r"C:\Users\dongdong.zhao\Desktop\tmp\0.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(ROI_res[1]), r"C:\Users\dongdong.zhao\Desktop\tmp\1.nii.gz")
        # sitk.WriteImage(sitk.GetImageFromArray(ROI_res[2]), r"C:\Users\dongdong.zhao\Desktop\tmp\2.nii.gz")
        # raise ValueError

        ROI_res = torch.nn.functional.interpolate(ROI_res, size=ROI_size, mode='nearest')

        ROI_res = ROI_res.squeeze().cpu().numpy()
        # ROI_res = (ROI_res > 0.1).astype(np.int8)
        tmp_ROI_res = np.argmax(ROI_res, axis=0)


        # artery[artery > threshold] = 1

        # artery = keep_maximum_volume(artery)
        # vein[vein > cfg.threshold] = 1
        # vein = keep_maximum_volume(vein)

        res0 = np.zeros_like(processed_ct)
        res1 = np.zeros_like(processed_ct)
        res2 = np.zeros_like(processed_ct)
        tmp_res = np.zeros_like(processed_ct)
        res0[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = ROI_res[0]
        res1[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = ROI_res[1]
        res2[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = ROI_res[2]
        tmp_res[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = tmp_ROI_res

        # res = np.asarray((res > cfg.threshold)).astype(np.uint8)
        # res = keep_maximum_volume(res)



        # 保存预测结果
        if not os.path.exists(os.path.join(cfg.test_save_path, patient_name)):
            os.makedirs(os.path.join(cfg.test_save_path, patient_name))
        sitk.WriteImage(sitk.GetImageFromArray(res0), os.path.join(cfg.test_save_path, patient_name, cfg.item + "_res0.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(res1), os.path.join(cfg.test_save_path, patient_name, cfg.item + "_res1.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(res2), os.path.join(cfg.test_save_path, patient_name, cfg.item + "_res2.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(tmp_res), os.path.join(cfg.test_save_path, patient_name, cfg.item + "_tmp.nii.gz"))

        # # 计算dice
        # label_path = os.path.join(patient_path, "PulmonaryVessels.nii.gz")
        #
        # if os.path.exists(label_path):
        #     label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        #     label_array[label_array > 1] = 0
        #     dice = (2 * res * label_array).sum() / (res.sum() + label_array.sum() + 1e-8)
        # else:
        #     dice = None
        #
        # return {patient_name: dice}


if __name__ == '__main__':
    # thead pool
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_time = time.time()
    patient_path_list = get_path(cfg.test_root_path)

    model, _ = model_zoo.create_model(cfg)


    model.load_state_dict(torch.load(cfg.used_ckpt)['model_state_dict'])
    model.to(cfg.device)
    model.eval()

    res_list = []

    for sub in patient_path_list:
        sub_result = predict(cfg, model, sub)
        print(sub_result)
        # patient_name = sub.split("\\")[-1]
        # res_list.append(sub_result[patient_name])

    print("time:{}".format(time.time() - start_time))
    print("mean dice:", sum(res_list) / len(res_list))



