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
        lung_mask = torch.Tensor(lung_mask).unsqueeze(0).unsqueeze(0)
        lung_mask = F.interpolate(lung_mask, size=processed_ct.shape, mode='nearest')
        lung_mask = lung_mask.squeeze().cpu().detach().numpy()
        bbox = get_boundaries_from_mask(lung_mask)
        ROI = processed_ct[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]


        # 单输出网络
        res = np.zeros_like(processed_ct)
        ROI_res= overlap_predict(net, ROI, patch_size=cfg.patch_size, rate=2)   # 预测结果



        res[bbox['zmin']:bbox['zmax'], bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']] = ROI_res
        # 重采样到原来大小
        res = torch.Tensor(res).unsqueeze(0).unsqueeze(0)
        res = F.interpolate(res, size=origial_size, mode='nearest')
        res = res.squeeze().cpu().detach().numpy()
        res = np.asarray((res > cfg.threshold)).astype(np.uint8)
        res = keep_maximum_volume(res)


        # WingsNet网络预测
        # res, res1 = overlap_predict(net, processed_ct, patch_size=cfg.patch_size, rate=2)   # 预测结果
        # # 重采样到原来大小
        # res = torch.Tensor(res).unsqueeze(0).unsqueeze(0)
        # res = F.interpolate(res, size=origial_size, mode='nearest')
        # res = res.squeeze().cpu().detach().numpy()
        # res = np.asarray((res > cfg.threshold)).astype(np.uint8)
        # # res = keep_maximum_volume(res)
        #
        # res1 = torch.Tensor(res1).unsqueeze(0).unsqueeze(0)
        # res1 = F.interpolate(res1, size=origial_size, mode='nearest')
        # res1 = res1.squeeze().cpu().detach().numpy()
        # res1 = np.asarray((res1 > cfg.threshold)).astype(np.uint8)
        #
        # res = np.logical_or(res, res1)
        # res = keep_maximum_volume(res)
        # # res = (res + res1) / 2
        # 保存预测结果
        res_itk = sitk.GetImageFromArray(res)
        if not os.path.exists(os.path.join(cfg.test_save_path, patient_name)):
            os.makedirs(os.path.join(cfg.test_save_path, patient_name))
        sitk.WriteImage(res_itk, os.path.join(cfg.test_save_path, patient_name, cfg.item + "_predict.nii.gz.nii.gz"))

        # 计算dice
        label_path = os.path.join(patient_path, "PulmonaryVessels.nii.gz")

        if os.path.exists(label_path):
            label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
            label_array[label_array > 0] = 1
            dice = (2 * res * label_array).sum() / (res.sum() + label_array.sum() + 1e-8)
        else:
            dice = None

        return {patient_name: dice}


if __name__ == '__main__':
    # thead pool
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    start_time = time.time()
    patient_path_list = get_path(cfg.test_root_path)

    model, opt = model_zoo.create_model(cfg)
    model.load_state_dict(torch.load(cfg.used_ckpt)['model_state_dict'])
    model.to(cfg.device)
    model.eval()

    res_list = []

    for sub in patient_path_list:
        sub_result = predict(cfg, model, sub)
        print(sub_result)
        patient_name = sub.split("\\")[-1]
        res_list.append(sub_result[patient_name])

    print("time:{}".format(time.time() - start_time))
    print("mean dice:", sum(res_list) / len(res_list))



