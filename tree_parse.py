import numpy as np
import os
from scipy import ndimage
import skimage.measure as measure
from skimage.morphology import skeletonize
from utils.utils import keep_maximum_volume
import SimpleITK as sitk
import itertools



def skeleton_parsing(skeleton):
    #separate the skeleton
    neighbor_filter = ndimage.generate_binary_structure(3, 3)  # 3*3*3的True 矩阵
    skeleton_filtered = ndimage.convolve(skeleton, neighbor_filter) * skeleton
    #distribution = skeleton_filtered[skeleton_filtered>0]
    #plt.hist(distribution)
    skeleton_parse = skeleton.copy()

    skeleton_parse[skeleton_filtered>3] = 0  # 交点处设为0

    con_filter = ndimage.generate_binary_structure(3, 3)
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    #remove small branches
    for i in range(num):
        a = cd[cd==(i+1)]
        if a.shape[0]<5:
            skeleton_parse[cd==(i+1)] = 0
    cd, num = ndimage.label(skeleton_parse, structure = con_filter)
    return skeleton_parse, cd, num

def tree_parsing_func(skeleton_parse, label, cd):
    #parse the airway tree
    edt, inds = ndimage.distance_transform_edt(1-skeleton_parse, return_indices=True)
    # tree_parsing = np.zeros(label.shape, dtype = np.uint16)
    tree_parsing = cd[inds[0,...], inds[1,...], inds[2,...]] * label
    return tree_parsing

def loc_trachea(tree_parsing, num):
    #find the trachea
    volume = np.zeros([num])
    for k in range(num):
        volume[k] = ((tree_parsing==(k+1)).astype(np.uint8)).sum()
    volume_sort = np.argsort(volume)
    #print(volume_sort)
    trachea = (volume_sort[-1]+1)
    return trachea

def adjacent_map(tree_parsing, num):
    #build the adjacency matric
    ad_matric = np.zeros((num, num), dtype=np.uint8)
    #i = 1
    for i in range(num):
        cd_cur = (tree_parsing==(i+1)).astype(np.uint8)
        xl, xr, yl, yr, zl, zr = find_bb_3D(cd_cur)
        cd_cur = cd_cur[xl:xr, yl:yr, zl:zr]
        #edt = ndimage.distance_transform_edt(1-cd_cur, return_indices=False)
        dilation_filter = ndimage.generate_binary_structure(3, 1)
        boundary = ndimage.binary_dilation(cd_cur, structure=dilation_filter).astype(cd_cur.dtype) - cd_cur
        adjacency = boundary*tree_parsing[xl:xr, yl:yr, zl:zr]
        adjacency_elements = np.unique(adjacency[adjacency>0])
        for j in range(len(adjacency_elements)):
            ad_matric[i,adjacency_elements[j]-1] = 1
    return ad_matric


def find_bb_3D(label):
    if len(label.shape) != 3:
        print("The dimension of input is not 3!")
        os._exit()
    sum_x = np.sum(label, axis=(1, 2))
    sum_y = np.sum(label, axis=(0, 2))
    sum_z = np.sum(label, axis=(0, 1))
    xf = np.where(sum_x)
    xf = xf[0]
    yf = np.where(sum_y)
    yf = yf[0]
    zf = np.where(sum_z)
    zf = zf[0]
    x_length = xf.max() - xf.min() + 1
    y_length = yf.max() - yf.min() + 1
    z_length = zf.max() - zf.min() + 1
    x1 = xf.min()
    y1 = yf.min()
    z1 = zf.min()

    cs = [x_length + 8, y_length + 8, z_length + 8]
    for j in range(3):
        if cs[j] > label.shape[j]:
            cs[j] = label.shape[j]
    # print(cs[0], x_length)
    # x_length, y_length, z_length, x1, y1, z1 = find_bb_3D(label2)
    cs = np.array(cs, dtype=np.uint16)
    size = label.shape
    xl = x1 - (cs[0] - x_length) // 2
    yl = y1 - (cs[1] - y_length) // 2
    zl = z1 - (cs[2] - z_length) // 2
    xr = xl + cs[0]
    yr = yl + cs[1]
    zr = zl + cs[2]
    if xl < 0:
        xl = 0
        xr = cs[0]
    if xr > size[0]:
        xr = size[0]
        xl = xr - cs[0]
    if yl < 0:
        yl = 0
        yr = cs[1]
    if yr > size[1]:
        yr = size[1]
        yl = yr - cs[1]
    if zl < 0:
        zl = 0
        zr = cs[2]
    if zr > size[2]:
        zr = size[2]
        zl = zr - cs[2]
    return xl, xr, yl, yr, zl, zr


def parent_children_map(ad_matric, trachea, num):
    # build the parent map and children map
    parent_map = np.zeros((num, num), dtype=np.uint8)
    children_map = np.zeros((num, num), dtype=np.uint8)
    generation = np.zeros((num), dtype=np.uint8)
    processed = np.zeros((num), dtype=np.uint8)

    processing = [trachea - 1]
    parent_map[trachea - 1, trachea - 1] = 1
    while len(processing) > 0:
        iteration = processing
        print("items in this iteration: ", iteration)
        processed[processing] = 1
        processing = []
        while len(iteration) > 0:
            cur = iteration.pop()
            children = np.where(ad_matric[cur, :] > 0)[0]
            for i in range(len(children)):
                cur_child = children[i]
                if parent_map[cur_child, :].sum() == 0:
                    parent_map[cur_child, cur] = 1
                    children_map[cur, cur_child] = 1
                    generation[cur_child] = generation[cur] + 1
                    processing.append(cur_child)
                else:
                    if generation[cur] + 1 == generation[cur_child]:
                        parent_map[cur_child, cur] = 1
                        children_map[cur, cur_child] = 1
    return parent_map, children_map, generation


def whether_refinement(parent_map, children_map, tree_parsing, num, trachea):
    witem = np.sum(parent_map, axis=1)
    witems = np.where(witem > 1)[0]
    child_num = np.sum(children_map, axis=1)
    problem1_loc = np.where(child_num == 1)[0]

    # First, fuse the parents of one child
    delete_ids = []
    if len(witems) > 0:
        for i in range(len(witems)):
            # print("item: ", witems[i], "parents: ", np.where(parent_map[witems[i],:]>0)[0])
            cur_witem = np.where(parent_map[witems[i], :] > 0)[0]
            for j in range(1, len(cur_witem)):
                tree_parsing[tree_parsing == (cur_witem[j] + 1)] = cur_witem[0] + 1
                if cur_witem[j] not in delete_ids:
                    delete_ids.append(cur_witem[j])

    # second, delete the alone child
    for i in range(len(problem1_loc)):
        cur_loc = problem1_loc[i]
        if cur_loc not in delete_ids:
            cur_child = np.where(children_map[cur_loc, :] == 1)[0][0]
            if cur_child not in delete_ids:
                tree_parsing[tree_parsing == (cur_child + 1)] = cur_loc + 1
                delete_ids.append(cur_child)

    # =============================================================================
    #     #Third, delete the wrong trachea blocks
    #     Tchildren = np.where(children_map[trachea-1,:]>0)[0]
    #     z_trachea = np.mean(np.where(cd==(trachea))[0])
    #     for i in range(len(Tchildren)):
    #         z_child = np.mean(np.where(cd==(Tchildren[i]+1))[0])
    #         if z_child > z_trachea:
    #             if Tchildren[i] not in delete_ids:
    #                 tree_parsing[tree_parsing==(Tchildren[i]+1)] = trachea
    #                 delete_ids.append(Tchildren[i])
    # =============================================================================

    if len(delete_ids) == 0:
        return False
    else:
        return True


def tree_refinement(parent_map, children_map, tree_parsing, num, trachea):
    witem = np.sum(parent_map, axis=1)
    witems = np.where(witem > 1)[0]
    if len(witems) > 0:
        for i in range(len(witems)):
            print("item: ", witems[i], "parents: ", np.where(parent_map[witems[i], :] > 0)[0])

    # print(np.where(children_map[160,:]>0)[0])
    child_num = np.sum(children_map, axis=1)
    problem1_loc = np.where(child_num == 1)[0]

    # First, fuse the parents of one child
    delete_ids = []
    if len(witems) > 0:
        for i in range(len(witems)):
            # print("item: ", witems[i], "parents: ", np.where(parent_map[witems[i],:]>0)[0])
            cur_witem = np.where(parent_map[witems[i], :] > 0)[0]
            for j in range(1, len(cur_witem)):
                tree_parsing[tree_parsing == (cur_witem[j] + 1)] = cur_witem[0] + 1
                if cur_witem[j] not in delete_ids:
                    delete_ids.append(cur_witem[j])

    # second, delete the only child
    for i in range(len(problem1_loc)):
        cur_loc = problem1_loc[i]
        if cur_loc not in delete_ids:
            cur_child = np.where(children_map[cur_loc, :] == 1)[0][0]
            if cur_child not in delete_ids:
                tree_parsing[tree_parsing == (cur_child + 1)] = cur_loc + 1
                delete_ids.append(cur_child)

    # =============================================================================
    #     #Third, delete the wrong trachea blocks
    #     Tchildren = np.where(children_map[trachea-1,:]>0)[0]
    #     z_trachea = np.mean(np.where(cd==(trachea))[0])
    #     for i in range(len(Tchildren)):
    #         z_child = np.mean(np.where(cd==(Tchildren[i]+1))[0])
    #         if z_child > z_trachea:
    #             if Tchildren[i] not in delete_ids:
    #                 tree_parsing[tree_parsing==(Tchildren[i]+1)] = trachea
    #                 delete_ids.append(Tchildren[i])
    # =============================================================================

    # delete the problematic blocks from the tree
    for i in range(num):
        if i not in delete_ids:
            move = len(np.where(np.array(delete_ids) < i)[0])
            tree_parsing[tree_parsing == (i + 1)] = i + 1 - move
    num = num - len(delete_ids)

    return tree_parsing, num



def tree_parse(mask_path, save_root_path):
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    main_label = keep_maximum_volume(mask)
    skeleton = skeletonize(main_label)
    skeleton_parse, cd, num = skeleton_parsing(skeleton)   # 返回值：分段后的细化（数值都为1），分段后的细化（不同段数值不一致），段个数
    tree_parsing = tree_parsing_func(skeleton_parse, main_label, cd)

    trachea = loc_trachea(tree_parsing, num)
    ad_matric = adjacent_map(tree_parsing, num)
    parent_map, children_map, generation = parent_children_map(ad_matric, trachea, num)
    while whether_refinement(parent_map, children_map, tree_parsing, num, trachea) is True:
        tree_parsing, num = tree_refinement(parent_map, children_map, tree_parsing, num, trachea)
        trachea = loc_trachea(tree_parsing, num)
        ad_matric = adjacent_map(tree_parsing, num)
        parent_map, children_map, generation = parent_children_map(ad_matric, trachea, num)

    sitk.WriteImage(sitk.GetImageFromArray(tree_parsing), os.path.join(save_root_path, "parse.nii.gz"))
    sitk.WriteImage(sitk.GetImageFromArray(skeleton), os.path.join(save_root_path, "skeleton.nii.gz"))

if __name__ == '__main__':
    mask_path = r"C:\dongdong.zhao\data\EXTACT_09\train_data\Annotation\CASE02.mhd"
    tree_parse(mask_path, save_root_path=r"C:\Users\dongdong.zhao\Desktop")
