import numpy as np
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



patient_list =              ['102',   '108',   '115',   '116',   '117',   '118',   '138',   '148',   '149',   '154',   '156',   '17',    '189',   '228',   '237',   '244',   '248',   '253',   '269',   '293',   '299',   '302',   '312',   '313',   '317',   '318',   '385',   '386',   '405',   '412',   '413']
Unet_ours_DiceBCE =         [0.750, 0.821, 0.841, 0.847, 0.830, 0.788, 0.810, 0.804, 0.758, 0.847, 0.800, 0.817, 0.825, 0.745, 0.813, 0.840, 0.739, 0.782, 0.791, 0.770, 0.833, 0.804, 0.792, 0.794, 0.825, 0.655, 0.789, 0.820, 0.707, 0.803, 0.830] # Unet_ours
WingsNet_DiceBCE =          [0.733, 0.831, 0.864, 0.862, 0.827, 0.800, 0.807, 0.776, 0.748, 0.852, 0.791, 0.822, 0.807, 0.739, 0.819, 0.823, 0.736, 0.762, 0.795, 0.775, 0.822, 0.817, 0.818, 0.797, 0.811, 0.646, 0.783, 0.796, 0.758, 0.803, 0.835]  # WingsNet
Unet_ours_Tversky =         [0.753, 0.816, 0.854, 0.832, 0.808, 0.748, 0.775, 0.769, 0.711, 0.826, 0.740, 0.790, 0.800, 0.745, 0.778, 0.830, 0.682, 0.774, 0.772, 0.731, 0.818, 0.817, 0.766, 0.783, 0.782, 0.612, 0.739, 0.766, 0.642, 0.742, 0.811]
Unet_ours_down3_Tversky =   [0.734, 0.809, 0.854, 0.833, 0.800, 0.747, 0.773, 0.772, 0.697, 0.829, 0.736, 0.784, 0.805, 0.750, 0.763, 0.832, 0.662, 0.781, 0.780, 0.712, 0.814, 0.827, 0.770, 0.780, 0.765, 0.594, 0.722, 0.742, 0.663, 0.750, 0.812]
mean_result0 = round(sum(Unet_ours_DiceBCE) / len(Unet_ours_DiceBCE), 3)
mean_result1 = round(sum(WingsNet_DiceBCE) / len(WingsNet_DiceBCE), 3)
mean_result2 = round(sum(Unet_ours_Tversky) / len(Unet_ours_Tversky), 3)
mean_result3 = round(sum(Unet_ours_down3_Tversky) / len(Unet_ours_down3_Tversky), 3)



bar_width = 0.2
index1 = np.arange(len(patient_list))
index2 = index1 + bar_width * 1
index3 = index1 + bar_width * 2
index4 = index1 + bar_width * 3
model = ("Unet_ours_DiceBCE=" + str(mean_result0),
         "WingsNet_DiceBCE=" + str(mean_result1),
         "Unet_ours_Tversky=" + str(mean_result2),
         "Unet_ours_down3_Tversky=" + str(mean_result3))

plt.figure(figsize=(16,9))

plt.bar(index1, height=Unet_ours_DiceBCE, width=bar_width, color='aqua', label=model[0])
plt.bar(index2, height=WingsNet_DiceBCE, width=bar_width, color='fuchsia', label=model[1])
plt.bar(index3, height=Unet_ours_Tversky, width=bar_width, color='green', label=model[2])
plt.bar(index4, height=Unet_ours_down3_Tversky, width=bar_width, color='maroon', label=model[3])


# 均值
plt.hlines(mean_result0, 0, len(patient_list), colors='aqua',linestyles='--')
plt.hlines(mean_result1, 0, len(patient_list), colors='fuchsia',linestyles='--')
plt.hlines(mean_result2, 0, len(patient_list), colors='green',linestyles='--')
plt.hlines(mean_result3, 0, len(patient_list), colors='maroon',linestyles='--')

plt.ylim((0.5, 1.05))
plt.legend(loc='upper right')
plt.xticks(index1 + bar_width/2, patient_list)
plt.ylabel("DICE")
plt.title("测试数据的dice评价指标")
plt.show()