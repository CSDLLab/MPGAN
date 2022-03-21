import numpy as np
from klib.baseio import *
from scipy.ndimage import filters as ndfilter
from klib.glib.DrawSimulationSWCModel import simulate3DTreeModel_dendrite,simulate3DTreeModel_axon, save_swc
import copy
import cv2 as cv
import multiprocessing as mp
from skimage import morphology
import time


# init parameter
data_type = np.uint16
internal_feature = True
external_feature = True
image_number = 6

# size of training data
size_x = 64
size_y = 64
channel = 32


# data dir
data_image_dir = 'data/simulator_data/sim_img/'
data_label_dir = 'data/simulator_data/sim_label/'
data_swc_dir = 'data/simulator_data/sim_swc/'

train_data_dir = 'data/simulator_data/data_divide/'
train_label_dir = 'data/simulator_data/label_divide/'

def main(img_num):
    np.random.seed()

    k = 1
    # size of fake neuron images
    MAX_BOX_Z = 256
    MAX_BOX_X = 256 * 2
    MAX_BOX_Y = 256

    MAX_BOX_WIDTH = [MAX_BOX_Z,MAX_BOX_X,MAX_BOX_Y]


    img_sim, label_sim, swc_data = simulate3DTreeModel_dendrite(MAX_BOX_WIDTH, internal_feature, external_feature,  data_type=data_type)
    # img_sim, label_sim, swc_data = simulate3DTreeModel_axon(MAX_BOX_WIDTH, internal_feature, external_feature,  data_type=data_type)
    SHAPE = img_sim.shape

    img_new = np.zeros([int(SHAPE[0] // 2), int(SHAPE[1]), int(SHAPE[2])], dtype=data_type)
    label_new = np.zeros([int(SHAPE[0] // 2), int(SHAPE[1]), int(SHAPE[2])], dtype=np.uint8)


    # adjust anisotropic resolutions  Z: 1->0.5
    for y in range(SHAPE[2]):
        img_temp = copy.deepcopy(img_sim[:, :, y:y + 1].reshape((SHAPE[0], SHAPE[1])))
        img_temp_1 = cv.resize(img_temp, (int(SHAPE[1]), int(SHAPE[0])//2))
        img_temp_2 = copy.deepcopy(img_temp_1.reshape((int(SHAPE[0] // 2), int(SHAPE[1]), 1)))
        img_new[:, :, y:y + 1] = copy.deepcopy(img_temp_2)

        label_temp = copy.deepcopy(label_sim[:, :, y:y + 1].reshape((SHAPE[0], SHAPE[1])))
        label_temp_1 = cv.resize(label_temp, (int(SHAPE[1]), int(SHAPE[0]//2)))
        label_temp_2 = copy.deepcopy(label_temp_1.reshape((int(SHAPE[0] // 2), int(SHAPE[1]), 1)))
        label_new[:, :, y:y + 1] = copy.deepcopy(label_temp_2)

    for i in range(swc_data.shape[0]):
        swc_data[i][4] = swc_data[i][4] / 2

    data_image_dir_tmp = data_image_dir + str(img_num + 1) + '.tif'
    data_label_dir_tmp = data_label_dir + str(img_num + 1) + '.gt.tif'
    data_swc_dir_tmp = data_swc_dir + str(img_num + 1) + '.swc'

    save_tif(img_new, data_image_dir_tmp, data_type)
    save_tif(label_new, data_label_dir_tmp, np.uint8)
    save_swc(data_swc_dir_tmp, swc_data)


    # make sure the branch is centered in the image
    c, a, b = img_new.shape
    skel = morphology.skeletonize_3d(label_new)
    skl_random_mask = np.random.randint(0,20,size=(c,a,b))
    skl_random = skl_random_mask * skel
    skl_random[skl_random!=1]=0
    skl_seed_pos = np.where(skl_random==1)

    for slice_num in range(skl_seed_pos[0].shape[0]):
        z = skl_seed_pos[0][slice_num]
        x = skl_seed_pos[1][slice_num]
        y = skl_seed_pos[2][slice_num]

        if z < channel//2 or z > c - channel//2 or x < size_x//2 or x > a - size_x//2 or y < size_y//2 or y > b - size_y//2:
            continue
        else:
            temp_image = img_new[z-channel//2:z + channel//2, x - size_x//2:x + size_x//2, y - size_y//2:y + size_y//2]
            temp_label = label_new[z-channel//2:z + channel//2, x - size_x//2:x + size_x//2, y - size_y//2:y + size_y//2]

            newname_fake = train_data_dir + str(img_num + 1) + '_' + str(k) + '.tif'
            newname_label = train_label_dir + str(img_num + 1) + '_' + str(k) + '.tif'

            save_tif(temp_image, newname_fake, data_type)
            save_tif(temp_label, newname_label, np.uint8)
            k = k + 1

    print('image ID: %d, number of training images: %d' % (img_num, k))
    time.sleep(1)


if __name__ == '__main__':
    # multiprocessing
    cpu_core_num = 5
    pool = mp.Pool(processes=cpu_core_num)  # we set cpu core is 4
    pool.map(main, range(0, image_number))

