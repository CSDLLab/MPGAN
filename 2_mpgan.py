import os
import sys
from shutil import rmtree
import random
import glob
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from klib.baseio import *
from klib.glib.Draw3DTools import *
from klib.image_history_buffer import ImageHistoryBuffer
from scipy.ndimage import filters as ndfilter
from klib.glib.DrawSimulationSWCModel import simulate3DTreeModel_dendrite
from skimage import morphology
import copy
import math
import cv2 as cv
# from pylab import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
model_num = 0

DATA_DIR = 'data'

MODE = 'test'
# MODE = 'train'

to_pre_train = False
to_restore = False


HISTORYBUFFER = 'True'
SHAPE = (32, 64, 64)
MAX_STEPS = 20000
BATCH_SIZE = 10

STEP_REFINER = 4
STEP_DISCRIMINATOR = 1

STEP_EVALUATE = 1

LAMBDA = 5
K = 0.1
LAMBDA_SU = 0.5


# =================================load data=============================================================================
# real data  and  fake data

data_type_ = np.uint16

real_image_dir = 'data/real_image_fmost/*.tif'

fake_image_dir = 'data/simulator_data/data_divide/*.tif'
fake_label_dir = 'data/simulator_data/label_divide/*.tif'


real_data = glob.glob(real_image_dir)
fake_data = glob.glob(fake_image_dir)
# test_image = glob.glob('/4T/liuchao/simGAN/test_image/*.tif')

real_data_num = int(len(real_data))
fake_data_num = int(len(fake_data))


def batch_real_new():
    real_file_names = glob.glob(real_image_dir)
    num_real_file_names = int(len(real_file_names))
    imgs = np.zeros(dtype=np.float32, shape=(BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1))
    for i in range(BATCH_SIZE):
        img = open_tif(real_file_names[random.randint(0, num_real_file_names) - 1])
        temp_data = np.reshape(img, [SHAPE[0], SHAPE[1], SHAPE[2], 1])
        imgs[i] = copy.deepcopy(temp_data)
    if data_type_ == np.uint16:
        imgs = np.sqrt(imgs) / 255 * 2 - 1
    else:
        imgs = (imgs) / 255 * 2 - 1
    return imgs


def batch_fake_new():
    fake_file_names = glob.glob(fake_image_dir)
    fake_label_names = glob.glob(fake_label_dir)
    num_fake_file_names = int(len(fake_file_names))

    imgs = np.zeros(dtype=np.float32, shape=(BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1))
    mask = np.zeros(dtype=np.float32, shape=(BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1))

    for i in range(BATCH_SIZE):
        k = random.randint(0, num_fake_file_names) - 1
        img = open_tif(fake_file_names[k])
        label = open_tif(fake_label_names[k])

        temp_data = np.reshape(img, [SHAPE[0], SHAPE[1], SHAPE[2], 1])
        temp_label = np.reshape(label, [SHAPE[0], SHAPE[1], SHAPE[2], 1])
        imgs[i] = copy.deepcopy(temp_data)
        mask[i] = copy.deepcopy(temp_label)

    if data_type_ == np.uint16:
        imgs = np.sqrt(imgs) / 255 * 2 - 1
    else:
        imgs = (imgs) / 255 * 2 - 1
    return imgs, mask


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)
        else:
            return tf.maximum(x, leak * x)


def resnet_block(input, filters=64, name="resnet"):
    with tf.variable_scope(name):
        layer = tf.layers.conv3d(input, filters, 3, 1, 'SAME')
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer = lrelu(layer)
        layer = tf.layers.conv3d(layer, filters, 3, 1, 'SAME')
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        return lrelu(tf.add(layer, input))


def define_discriminator(x, reuse=False, name="discriminator"):
    with tf.variable_scope(name, reuse=reuse):
        layer = tf.layers.conv3d(x, 64, 3, 2, 'SAME')
        layer = tf.layers.conv3d(layer, 128, 3, 2, 'SAME')
        layer = tf.nn.max_pool3d(
            layer, (1, 3, 3, 3, 1), (1, 1, 1, 1, 1), 'SAME')
        layer = tf.layers.conv3d(layer, 128, 3, 2, 'SAME')
        layer = tf.layers.conv3d(layer, 64, 1, 1, 'SAME')
        output = tf.layers.conv3d(layer, 1, 1, 1, 'SAME')

        return output


def define_refiner(x, reuse=False, name="generator"):
    with tf.variable_scope(name, reuse=reuse):
        # conv3d layer
        layer = tf.layers.conv3d(x, 32, 5, 1, 'SAME')
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer = lrelu(layer)
        layer = tf.layers.conv3d(layer, 32, 3, 1, 'SAME')
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer = lrelu(layer)
        layer = tf.layers.conv3d(layer, 32, 3, 1, 'SAME')
        layer = tf.contrib.layers.batch_norm(layer, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
        layer = lrelu(layer)

        # resnet block
        layer = resnet_block(layer, 32, name='res_1')
        layer = resnet_block(layer, 32, name='res_2')
        layer = resnet_block(layer, 32, name='res_3')
        layer = resnet_block(layer, 32, name='res_4')

        layer = tf.layers.conv3d(layer, 32, 3, 1, 'SAME')
        layer = lrelu(layer)
        logits = tf.layers.conv3d(layer, 1, 1, 1, 'SAME')

        # tanh layer
        output = tf.nn.tanh(logits)
        return output


def define_graph(LOSS):
    rv = {}

    # input
    rv['real_img'] = tf.placeholder(tf.float32, [BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1], name="input_real")
    rv['fake_img'] = tf.placeholder(tf.float32, [BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1], name="input_fake")
    rv['fake_ths'] = tf.placeholder(tf.float32, [BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1], name="input_fake_mask")
    rv['refined_ths'] = tf.placeholder(tf.float32, [BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1],
                                       name="input_refined_mask")
    rv['fake_pool'] = tf.placeholder(tf.float32, [None, SHAPE[0], SHAPE[1], SHAPE[2], 1], name="fake_pool")

    rv['training_State'] = tf.placeholder(tf.bool)
    rv['K'] = tf.placeholder(tf.float32)
    rv['global_step'] = tf.Variable(0, name="global_step", trainable=False)

    # graph
    rv['r_output'] = define_refiner(rv['fake_img'], name="g_f2r")
    rv['d_real_output'] = define_discriminator(rv['real_img'], name="d")
    rv['d_refine_output'] = define_discriminator(rv['r_output'], reuse=True, name="d")
    rv['fake_pool_rec'] = define_discriminator(rv['fake_pool'], reuse=True, name="d")

    # D loss
    rv['d_loss_real'] = tf.reduce_mean(tf.squared_difference(rv['d_real_output'], 1))
    rv['d_loss_refine'] = tf.reduce_mean(tf.square(rv['fake_pool_rec']))
    rv['d_loss'] = (rv['d_loss_real'] + rv['d_loss_refine'])

    # R loss
    rv['r_loss_realism'] = tf.reduce_mean(tf.squared_difference(rv['d_refine_output'], 1))
    

    if LOSS == 'SimGAN':
        rv['r_loss_reg'] = LAMBDA * tf.reduce_mean(tf.abs(rv['fake_img'] - rv['r_output']))
        rv['r_loss'] = rv['r_loss_reg']  + rv['r_loss_realism']
        rv['r_loss_su'] = 0
        rv['r_loss_recon']=0
    if LOSS == 'MPGAN':
        if data_type_ == np.uint16:
            x_rescale = ((rv['fake_img'] + 1) / 2 * 255) ** 2
            x_refiner_rescale = ((rv['r_output'] + 1) / 2 * 255) ** 2
        else:
            x_rescale = ((rv['fake_img'] + 1) / 2 * 255) 
            x_refiner_rescale = ((rv['r_output'] + 1) / 2 * 255) 
        x_label = 1 / (1 + tf.exp(-rv['K'] * (x_rescale - rv['fake_ths'])))
        x_refiner_label = 1 / (1 + tf.exp(-rv['K'] * (x_refiner_rescale - rv['refined_ths'])))

        rv['r_loss_su'] = LAMBDA_SU * tf.reduce_mean(tf.abs(x_label - x_refiner_label))
        rv['r_loss_recon'] = LAMBDA * tf.reduce_mean(tf.abs(rv['fake_img'] - rv['r_output']))

        rv['r_loss_reg'] = rv['r_loss_su'] + rv['r_loss_recon']
        rv['r_loss'] = rv['r_loss_realism'] + rv['r_loss_reg']# + rv['r_loss_cyc']

    # 参数
    all_variables = tf.trainable_variables()
    r2f_variables = [var for var in all_variables if 'g_f2r' in var.name]
    d_variables = [var for var in all_variables if 'd' in var.name]
    rv['learning_rate'] = tf.train.exponential_decay(learning_rate=1e-4, global_step=rv['global_step'],
                                                     decay_steps=1000, decay_rate=0.9, staircase=True)

    rv['f2r_train_op'] = tf.train.AdamOptimizer(rv['learning_rate'], beta1=0.5).minimize(rv['r_loss'],var_list=r2f_variables)
    rv['d_train_op'] = tf.train.AdamOptimizer(rv['learning_rate'], beta1=0.5).minimize(rv['d_loss'],var_list=d_variables)

    # summary
    tf.summary.scalar('r_loss', rv['r_loss'])
    tf.summary.scalar('d_loss', rv['d_loss'])
    tf.summary.scalar('d_loss_real', rv['d_loss_real'])
    tf.summary.scalar('d_loss_refine', rv['d_loss_refine'])
    tf.summary.scalar('r_loss_realism', rv['r_loss_realism'])
    tf.summary.scalar('r_loss_reg', rv['r_loss_reg'])

    if data_type_ == np.uint16:
        real_show = tf.minimum(((rv['real_img'] + 1) / 2 * 255)**2,500)/500*255
        original_show = tf.minimum(((rv['fake_img'] + 1) / 2 * 255)**2,500)/500*255
        refined_show = tf.minimum(((rv['r_output'] + 1) / 2 * 255)**2,500)/500*255
    else:
        real_show = (rv['real_img'] + 1) / 2 * 255
        original_show = (rv['fake_img'] + 1) / 2 * 255
        refined_show = (rv['r_output'] + 1) / 2 * 255


    real_iamge = tf.cast(real_show, tf.uint8)
    original = tf.cast(original_show, tf.uint8)
    refined = tf.cast(refined_show, tf.uint8)

    if LOSS == 'SimGAN':
        original_SU = tf.cast((original), tf.uint8)
        refined_SU = tf.cast((refined), tf.uint8)
    if LOSS == 'MPGAN':
        original_SU = tf.cast((x_label * 255), tf.uint8)
        refined_SU = tf.cast((x_refiner_label * 255), tf.uint8)

    for i in range(SHAPE[0]):
        tf.summary.image('depth_' + str(i), tf.stack(
            [real_iamge[0, i, :, :, :],original[0, i, :, :, :], refined[0, i, :, :, :], original_SU[0, i, :, :, :], refined_SU[0, i, :, :, :]]),
                         max_outputs=5)

    rv['summary'] = tf.summary.merge_all()
    return rv


# MP mask
def image_mp(fake_img, refined_img, mask):
    if data_type_ == np.uint16:
        fake_img_temp = ((fake_img + 1) / 2 * 255) ** 2
        refined_img_temp = ((refined_img + 1) / 2 * 255) ** 2
    else:
        fake_img_temp = ((fake_img + 1) / 2 * 255) 
        refined_img_temp = ((refined_img + 1) / 2 * 255) 

    fake_ths = np.ones(dtype=np.float32, shape=(BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1))
    refined_ths = np.ones(dtype=np.float32, shape=(BATCH_SIZE, SHAPE[0], SHAPE[1], SHAPE[2], 1))

    for image_idx in range(BATCH_SIZE):

        fake_bg = fake_img_temp[mask == 0]
        fake_bg_len = len(fake_bg)
        fake_bg_sort = np.sort(fake_bg, axis=0)
        th = fake_bg_sort[int(fake_bg_len*0.95)]
        fake_ths[image_idx] = fake_ths[image_idx] * th

        refined_bg = refined_img_temp[mask == 0]
        refined_bg_len = len(refined_bg)
        refined_bg_sort = np.sort(refined_bg, axis=0)
        
        th = refined_bg_sort[int(refined_bg_len*0.95)]

        if data_type_ == np.uint16:
            data_type_th = 1000
        else:
            data_type_th = 250


        if th>data_type_th:
            th = fake_bg_sort[int(fake_bg_len*0.98)]
        if th<1:
            th = 1
        refined_ths[image_idx] = refined_ths[image_idx] * th

    return fake_ths, refined_ths


def pre_train(LOSS):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    MODEL_DIR = os.path.join(DATA_DIR, 'model/model_' + str(model_num))
    LOG_DIR = os.path.join(DATA_DIR, 'log/log_' + str(model_num))

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    g = define_graph(LOSS)
    with tf.Session() as s:
        s.run(tf.global_variables_initializer())  
        writer = tf.summary.FileWriter(LOG_DIR, s.graph)
        saver = tf.train.Saver(max_to_keep=10)
        for step in range(501):
            print('------pre-------', step)
            print(LOSS)
            real_img = batch_real_new()
            fake_img, mask = batch_fake_new()

            s.run(g['f2r_train_op'],feed_dict={g['fake_img']: fake_img})
            if step % 1 == 0:
                r_loss = s.run(g['r_loss'],feed_dict={g['fake_img']: fake_img})
                print('r_loss:', r_loss)
            if step % 50 == 0:
                saver.save(s, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=step + 1)


def train(LOSS):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    MODEL_DIR = os.path.join(DATA_DIR, 'model/model_' + str(model_num))
    LOG_DIR = os.path.join(DATA_DIR, 'log/log_' + str(model_num))

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    if os.path.exists(LOG_DIR):
        rmtree(LOG_DIR)
        os.mkdir(LOG_DIR)

    image_history_buffer = ImageHistoryBuffer((0, 32, 64, 64, 1), BATCH_SIZE * 100, BATCH_SIZE)
    g = define_graph(LOSS)


    with tf.Session() as s:
        s.run(tf.global_variables_initializer())  
        writer = tf.summary.FileWriter(LOG_DIR, s.graph)
        saver = tf.train.Saver(max_to_keep=10)

        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(MODEL_DIR)
            saver.restore(s, chkpt_fname)

        for step in range(MAX_STEPS):
            print('----------------', step)

            real_img = batch_real_new()
            fake_img, mask = batch_fake_new()

            
            # init real_ths, fake_ths
            refined_image = s.run(g['r_output'],feed_dict={g['fake_img']: fake_img})
            fake_ths, refined_ths = image_mp(fake_img, refined_image, mask)

            # train r
            for step_r in range(STEP_REFINER):
                _, refined_image = s.run([g['f2r_train_op'], g['r_output']],feed_dict={g['fake_img']: fake_img, g['real_img']: real_img, g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})

            # train d
            for step_d in range(STEP_DISCRIMINATOR):
                if HISTORYBUFFER == 'True':
                    # use a history of refined images
                    half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                    image_history_buffer.add_to_image_history_buffer(refined_image)
                    if len(half_batch_from_image_history):
                        refined_image[:BATCH_SIZE // 2] = half_batch_from_image_history
                s.run(g['d_train_op'], feed_dict={g['real_img']: real_img, g['fake_pool']: refined_image})

            if step % STEP_EVALUATE == 0:
                d_loss,r_loss,summary = s.run([g['d_loss'],g['r_loss'],g['summary']], feed_dict={g['real_img']: real_img, g['fake_img']: fake_img,g['fake_pool']: refined_image,g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
                r_loss_realism,r_loss_reg = s.run([g['r_loss_realism'],g['r_loss_reg']], feed_dict={g['fake_img']: fake_img, g['real_img']: real_img ,g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
                print('d_loss:', d_loss)
                print('r_loss:', r_loss)
                print('r_loss_realism:',r_loss_realism)
                print('r_loss_reg:',r_loss_reg)
                writer.add_summary(summary, step)
            if step % 10 == 0:
                saver.save(s, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=step + 1)
            print('----------------')



def test():
    np.random.seed()
    with tf.Session() as s:
        MODEL_DIR = os.path.join(DATA_DIR, 'model/model_' + str(model_num))
        REFINED_DIR = os.path.join(DATA_DIR, ' ')

        if not os.path.exists(REFINED_DIR):
            os.mkdir(REFINED_DIR)

        g = define_graph(LOSS='MPGAN')
        saver = tf.train.Saver()
        saver.restore(s, tf.train.latest_checkpoint(MODEL_DIR))

        internal_feature = True
        external_feature = True

        MAX_BOX_Z = 256
        MAX_BOX_X = 256 * 2
        MAX_BOX_Y = 256

        MAX_BOX_WIDTH = [MAX_BOX_Z,MAX_BOX_X,MAX_BOX_Y]

        test_image_num = 1
        cut_image_num = 1
        for img_id in range(test_image_num):
            img, label, swc_data = simulate3DTreeModel_dendrite(MAX_BOX_WIDTH, internal_feature, external_feature, data_type=data_type_)

            img_SHAPE = img.shape
            img_new = np.zeros([int(img_SHAPE[0] // 2), int(img_SHAPE[1]), int(img_SHAPE[2])], dtype=data_type_)
            label_new = np.zeros([int(img_SHAPE[0] // 2), int(img_SHAPE[1]), int(img_SHAPE[2])], dtype=np.uint8)

            # Z,X,Y
            for y in range(img_SHAPE[2]):
                img_temp = copy.deepcopy(img[:, :, y:y + 1].reshape((img_SHAPE[0], img_SHAPE[1])))
                img_temp_1 = cv.resize(img_temp, (int(img_SHAPE[1]), int(img_SHAPE[0])//2))
                img_temp_2 = copy.deepcopy(img_temp_1.reshape((int(img_SHAPE[0] // 2), int(img_SHAPE[1]), 1)))
                img_new[:, :, y:y + 1] = copy.deepcopy(img_temp_2)

                label_temp = copy.deepcopy(label[:, :, y:y + 1].reshape((img_SHAPE[0], img_SHAPE[1])))
                label_temp_1 = cv.resize(label_temp, (int(img_SHAPE[1]), int(img_SHAPE[0]//2)))
                kernel = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).astype(np.uint8)
                label_temp_dilate = cv.dilate(label_temp_1, kernel, iterations=1)
                label_temp_2 = copy.deepcopy(label_temp_dilate.reshape((int(img_SHAPE[0] // 2), int(img_SHAPE[1]), 1)))
                label_new[:, :, y:y + 1] = copy.deepcopy(label_temp_2)


            image_dir = 'data/train_data/fake_image/' + str(img_id + 1) + '.tif'
            label_dir = 'data/train_data/fake_label/' + str(img_id + 1) + '.gt.tif'
            refined_dir = 'data/train_data/refined_image/' + str(img_id + 1) + '.ref.tif'

            save_tif(img_new, image_dir, data_type_)
            save_tif(label_new, label_dir, np.uint8)

            if data_type_ == np.uint8:
                img_rescale = img_new / 255 * 2 - 1
            else:
                img_rescale = np.sqrt(img_new) / 255 * 2 - 1

            d, h, w = img_rescale.shape
            steps = 16
            batch = np.zeros(dtype=np.float32, shape=[BATCH_SIZE, *SHAPE, 1])
            cnt = 0

            refined_list = []
            refined_image = - np.ones_like(img_rescale, dtype=np.float32)

            for k in range(0, d - 16, steps):
                for j in range(0, h - 32, steps * 2):
                    for i in range(0, w - 32, steps * 2):
                        patch = img_rescale[k:k + SHAPE[0], j:j + SHAPE[1], i:i + SHAPE[2]]
                        r_d, r_h, r_w = patch.shape
                        batch[cnt, :r_d, :r_h, :r_w] = patch.reshape(r_d, r_h, r_w, 1).astype(np.float32)
                        cnt += 1
                        if cnt == BATCH_SIZE:
                            refined = s.run(g['r_output'], feed_dict={g['real_img']: batch, g['fake_img']: batch, g['training_State']: False})
                            for ref in refined:
                                refined_list.append(ref.reshape(SHAPE))
                            cnt = 0
            if cnt > 0:
                refined = s.run(g['r_output'],
                                feed_dict={g['real_img']: batch, g['fake_img']: batch, g['training_State']: False})
                for ref in refined:
                    refined_list.append(ref.reshape(SHAPE))

            index = 0
            for k in range(0, d - 16, steps):
                for j in range(0, h - 32, steps * 2):
                    for i in range(0, w - 32, steps * 2):
                        shape = refined_image[k:k + SHAPE[0], j:j + SHAPE[1], i:i + SHAPE[2]].shape
                        ref_img = refined_list[index]
                        refined_image[k + 8:k + SHAPE[0] - 8, j + 16:j + SHAPE[1] - 16, i + 16:i + SHAPE[2] - 16] = \
                            ref_img[8:shape[0] - 8, 16:shape[1] - 16, 16:shape[2] - 16]
                        index += 1

            if data_type_ == np.uint8:
                refined_image = ((refined_image + 1) / 2 * 255)
            else:
                refined_image = ((refined_image + 1) / 2 * 255) ** 2
                refined_image[refined_image > 16000] = 0

            save_tif(refined_image, refined_dir, data_type_)
            

            c, a, b = img_new.shape

            # skl
            skel = morphology.skeletonize_3d(label_new)
            skl_random_mask = np.random.randint(0,20,size=(c,a,b))
            skl_random = skl_random_mask * skel
            skl_random[skl_random!=1]=0
            skl_seed_pos = np.where(skl_random==1)

            size_x = 64
            size_y = 64
            channel = 32
            steps = 16


            for slice_num in range(skl_seed_pos[0].shape[0]):
                z = skl_seed_pos[0][slice_num]
                x = skl_seed_pos[1][slice_num]
                y = skl_seed_pos[2][slice_num]

                if z < channel//2 or z > c - channel//2 or x < size_x//2 or x > a - size_x//2 or y < size_y//2 or y > b - size_y//2:
                    continue
                else:
                    temp_refined_data = refined_image[z-channel//2:z + channel//2, x - size_x//2:x + size_x//2, y - size_y//2:y + size_y//2]
                    temp_image = img_new[z-channel//2:z + channel//2, x - size_x//2:x + size_x//2, y - size_y//2:y + size_y//2]
                    temp_label = label_new[z-channel//2:z + channel//2, x - size_x//2:x + size_x//2, y - size_y//2:y + size_y//2]
                    
                    newname_refined_data = 'data/train_data/data_divide_gan/' + str(img_id + 1) + '_' + str(cut_image_num)  + '.tif'
                    newname_fake_data = 'data/train_data/data_divide/' + str(img_id + 1) + '_'+ str(cut_image_num) + '.tif'
                    newname_label = 'data/train_data/label_divide/' + str(img_id + 1) + '_' + str(cut_image_num) + '.tif'

                    save_tif(temp_refined_data, newname_refined_data, data_type_)
                    save_tif(temp_image, newname_fake_data, data_type_)
                    save_tif(temp_label, newname_label, np.uint8)

                    cut_image_num = cut_image_num + 1
            print(cut_image_num)
            cut_image_num=1
            
            


if __name__ == '__main__':
    if MODE == 'train':
        if to_pre_train:
            pre_train(LOSS='SimGAN')
        train(LOSS='MPGAN')
    if MODE == 'test':
        test()

