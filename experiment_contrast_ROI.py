import numpy as np
import matplotlib.pyplot as plt

# 这两行代码解决 plt 中文显示的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def get_list(path):
    res_dict = {}
    res_list = []
    with open(path, 'r') as f:
        text = f.readlines()
        for sub in text:
            sub.strip('\n')
            k = sub.split(':')[0]
            v = sub.replace(' ', '').split(':')[-1].strip('\n')
            res_dict[k] = float(v)
            res_list.append(float(v))
    return res_dict, res_list

patient_list =            [102,   108,   115,   116,   117,   118,   138,   148,   149,   154,   156,   17,    189,   228,   237,   244,   248,   253]
Unet_ours_DiceBCE =       [0.761, 0.718, 0.822, 0.793, 0.727, 0.783, 0.742, 0.727, 0.753, 0.816, 0.729, 0.780, 0.753, 0.648, 0.755, 0.791, 0.648, 0.718]  # Unet_ours
WingsNet_DiceBCE =        [0.761, 0.720, 0.831, 0.793, 0.734, 0.780, 0.737, 0.733, 0.749, 0.814, 0.714, 0.775, 0.748, 0.678, 0.751, 0.718, 0.646, 0.703]  # WingsNet
Unet_ours_Tversky =       [0.751, 0.712, 0.845, 0.766, 0.726, 0.733, 0.704, 0.731, 0.703, 0.784, 0.655, 0.756, 0.702, 0.637, 0.711, 0.723, 0.596, 0.681]
Unet_ours_down3_Tversky = [0.736, 0.709, 0.845, 0.770, 0.725, 0.739, 0.710, 0.720, 0.700, 0.782, 0.655, 0.754, 0.699, 0.637, 0.705, 0.716, 0.589, 0.679]
graph_cuts =              [0.777, 0.659, 0.645, 0.793, 0.743, 0.792, 0.770, 0.568, 0.764, 0.822, 0.673, 0.728, 0.752, 0.416, 0.795, 0.813, 0.749, 0.658]



mean_result0 = round(sum(Unet_ours_DiceBCE) / len(Unet_ours_DiceBCE), 3)
mean_result1 = round(sum(WingsNet_DiceBCE) / len(WingsNet_DiceBCE), 3)
mean_result2 = round(sum(Unet_ours_Tversky) / len(Unet_ours_Tversky), 3)
mean_result3 = round(sum(Unet_ours_down3_Tversky) / len(Unet_ours_down3_Tversky), 3)
mean_result4 = round(sum(graph_cuts) / len(graph_cuts), 3)


bar_width = 0.15
index1 = np.arange(len(patient_list))
index2 = index1 + bar_width * 1
index3 = index1 + bar_width * 2
index4 = index1 + bar_width * 3
index5 = index1 + bar_width * 4

model = ("Unet_ours_DiceBCE=" + str(mean_result0),
         "WingsNet_DiceBCE=" + str(mean_result1),
         "Unet_ours_Tversky=" + str(mean_result2),
         "Unet_ours_down3_Tversky=" + str(mean_result3),
         "graphCuts=" + str(mean_result4))

plt.figure(figsize=(16,9))

plt.bar(index1, height=Unet_ours_DiceBCE, width=bar_width, color='aqua', label=model[0])
plt.bar(index2, height=WingsNet_DiceBCE, width=bar_width, color='fuchsia', label=model[1])
plt.bar(index3, height=Unet_ours_Tversky, width=bar_width, color='green', label=model[2])
plt.bar(index4, height=Unet_ours_down3_Tversky, width=bar_width, color='maroon', label=model[3])
plt.bar(index5, height=graph_cuts, width=bar_width, color='black', label=model[4])


# 均值
plt.hlines(mean_result0, 0, len(patient_list), colors='aqua',linestyles='--')
plt.hlines(mean_result1, 0, len(patient_list), colors='fuchsia',linestyles='--')
plt.hlines(mean_result2, 0, len(patient_list), colors='green',linestyles='--')
plt.hlines(mean_result3, 0, len(patient_list), colors='maroon',linestyles='--')
plt.hlines(mean_result4, 0, len(patient_list), colors='black',linestyles='--')

plt.ylim((0.5, 0.9))
plt.legend(loc='upper right')
plt.xticks(index1 + bar_width/2, patient_list)
plt.ylabel("DICE")
plt.title("测试数据的dice评价指标")
plt.show()
