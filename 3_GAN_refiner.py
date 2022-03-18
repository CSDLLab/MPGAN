import os
import sys
from shutil import rmtree
import random
import glob
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from klib.baseio import open_tif, save_tif
from klib.simulator import simulator, gaussianNoisyAddGray3D
from klib.image_history_buffer import ImageHistoryBuffer
import copy
import math
import cv2 as cv

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_num = 2

DATA_DIR = '/4T/liuchao/simGAN'

MODE = 'test'
# MODE = 'train'
to_pre_train = False

to_restore = True

# LOSS = 'simGAN'
# LOSS = 'SU'
# LOSS = 'SU_new'

HISTORYBUFFER = 'True'
# HISTORYBUFFER = 'False'

SHAPE = (16, 64, 64)
MAX_STEPS = 20000
BATCH_SIZE = 10

STEP_REFINER = 2
STEP_DISCRIMINATOR = 1

STEP_EVALUATE = 1

LAMBDA = 10

K = 0.2
LAMBDA_SU = 1
# LAMBDA_CYC = 10

# =================================load data=============================================================================
# real data  and  fake data

real_image_dir = '/4T/liuchao/simGAN/real_image_bigneuron/*.tif'
fake_image_dir = '/4T/liuchao/journal/m_2/data_divide/*.tif'
fake_label_dir = '/4T/liuchao/journal/m_2/label_divide/*.tif'

real_data = glob.glob(real_image_dir)
fake_data = glob.glob(fake_image_dir)


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
    imgs = np.sqrt(imgs) / 255 * 2 - 1
    # imgs = imgs / 255 * 2 - 1
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

        # fake_fg = temp_data[temp_label == 1]
        # fake_fg_1 = np.sort(fake_fg, axis=0)
        # fake_fg_len = len(fake_fg_1)
        # th = fake_fg_1[int(fake_fg_len * 0.2)]
        # ths[i] = ths[i] * th
        # print(ths[i])
    imgs = np.sqrt(imgs) / 255 * 2 - 1
    # imgs = imgs / 255 * 2 - 1
    return imgs, mask


def lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
    with tf.variable_scope(name):
        if alt_relu_impl:
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            # lrelu = 1/2 * (1 + leak) * x + 1/2 * (1 - leak) * |x|
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
        # 卷积层
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

        # 归到 -1 到 1之间
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
    rv['r_output'] = define_refiner(rv['fake_img'], name="g_r2f")
    # rv['fake_img_'] = define_refiner(rv['r_output'], name="g_f2r")
    rv['d_real_output'] = define_discriminator(rv['real_img'], name="d")
    rv['d_refine_output'] = define_discriminator(rv['r_output'], reuse=True, name="d")
    rv['fake_pool_rec'] = define_discriminator(rv['fake_pool'], reuse=True, name="d")

    # smooth = 0.1
    # D loss
    # rv['d_loss_real'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rv['d_real_logits'],
    #                                                                            labels=tf.ones_like(
    #                                                                                rv['d_real_logits']) * (1 - smooth)))
    # rv['d_loss_refine'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rv['d_refine_logits'],
    #                                                                              labels=tf.zeros_like(
    #                                                                                  rv['d_refine_logits']) * (
    #                                                                                             1 - smooth)))
    # rv['d_loss'] = rv['d_loss_real'] + rv['d_loss_refine']
    rv['d_loss_real'] = tf.reduce_mean(tf.squared_difference(rv['d_real_output'], 1))
    rv['d_loss_refine'] = tf.reduce_mean(tf.square(rv['fake_pool_rec']))
    rv['d_loss'] = (rv['d_loss_real'] + rv['d_loss_refine'])

    # R loss
    rv['r_loss_realism'] = tf.reduce_mean(tf.squared_difference(rv['d_refine_output'], 1))
    # rv['r_loss_realism'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=rv['d_refine_logits'],labels=tf.ones_like(rv['d_refine_logits']) * (1 - smooth)))
    # rv['r_loss_cyc'] = LAMBDA_CYC * tf.reduce_mean(tf.abs(rv['fake_img']-rv['fake_img_']))

    if LOSS == 'simGAN':
        rv['r_loss_reg'] = LAMBDA * tf.reduce_mean(tf.abs(rv['fake_img'] - rv['r_output']))
        rv['r_loss'] = rv['r_loss_reg']  + rv['r_loss_realism']
    if LOSS == 'SU':
        # x_rescale = ((rv['fake_img'] + 1) / 2 * 255) ** 2
        # x_refiner_rescale = ((rv['r_output'] + 1) / 2 * 255) ** 2
        # x_label = 1 / (1 + tf.exp(-rv['K'] * (x_rescale - th)))
        # x_refiner_label = 1 / (1 + tf.exp(-rv['K'] * (x_refiner_rescale - th)))

        x_label = 1 / (1 + tf.exp(-rv['K'] * (rv['fake_img'] - (tf.sqrt(rv['fake_ths']) / 255 * 2 - 1))))
        x_refiner_label = 1 / (1 + tf.exp(-rv['K'] * (rv['r_output'] - (tf.sqrt(rv['refined_ths']) / 255 * 2 - 1))))
        rv['r_loss_reg'] = LAMBDA_SU * tf.reduce_mean(tf.abs(x_label - x_refiner_label))
        rv['r_loss'] = rv['r_loss_realism'] + rv['r_loss_reg']
    if LOSS == 'SU_new':
        x_rescale = ((rv['fake_img'] + 1) / 2 * 255) ** 2
        x_refiner_rescale = ((rv['r_output'] + 1) / 2 * 255) ** 2
        x_label = 1 / (1 + tf.exp(-rv['K'] * (x_rescale - rv['fake_ths'])))
        x_refiner_label = 1 / (1 + tf.exp(-rv['K'] * (x_refiner_rescale - rv['refined_ths'])))

        # x_label = 1 / (1 + tf.exp(-rv['K'] * (rv['fake_img']- (tf.sqrt(rv['fake_ths']) / 255 * 2 - 1))))
        # x_refiner_label = 1 / (1 + tf.exp(-rv['K'] * (rv['r_output']- (tf.sqrt(rv['fake_ths']) / 255 * 2 - 1))))
        rv['r_loss_reg'] = LAMBDA_SU * tf.reduce_mean(tf.abs(x_label - x_refiner_label)) + LAMBDA * tf.reduce_mean(tf.abs(rv['fake_img'] - rv['r_output']))
        rv['r_loss'] = rv['r_loss_realism'] + rv['r_loss_reg'] # + rv['r_loss_cyc']

    # 参数
    all_variables = tf.trainable_variables()
    r2f_variables = [var for var in all_variables if 'g_r2f' in var.name]
    # f2r_variables = [var for var in all_variables if 'g_f2r' in var.name]
    d_variables = [var for var in all_variables if 'd' in var.name]
    rv['learning_rate'] = tf.train.exponential_decay(learning_rate=1e-4, global_step=rv['global_step'],
                                                     decay_steps=1000, decay_rate=0.9, staircase=True)

    rv['r2f_train_op'] = tf.train.AdamOptimizer(rv['learning_rate'], beta1=0.5).minimize(rv['r_loss'],var_list=r2f_variables)
    # rv['f2r_train_op'] = tf.train.AdamOptimizer(rv['learning_rate'], beta1=0.5).minimize(rv['r_loss_cyc'],var_list=f2r_variables)
    rv['d_train_op'] = tf.train.AdamOptimizer(rv['learning_rate'], beta1=0.5).minimize(rv['d_loss'],var_list=d_variables)

    # summary
    tf.summary.scalar('r_loss', rv['r_loss'])
    tf.summary.scalar('d_loss', rv['d_loss'])
    tf.summary.scalar('d_loss_real', rv['d_loss_real'])
    tf.summary.scalar('d_loss_refine', rv['d_loss_refine'])
    tf.summary.scalar('r_loss_realism', rv['r_loss_realism'])
    tf.summary.scalar('r_loss_reg', rv['r_loss_reg'])
    # tf.summary.scalar('Learning Rate', rv['learning_rate'])
    # tf.summary.scalar('K', rv['K'])

    # real_show = tf.minimum(((rv['real_img'] + 1) / 2 * 255)**2,500)/500*255
    # original_show = tf.minimum(((rv['fake_img'] + 1) / 2 * 255)**2,500)/500*255
    # refined_show = tf.minimum(((rv['r_output'] + 1) / 2 * 255)**2,500)/500*255
    real_show = (rv['real_img'] + 1) / 2 * 255
    original_show = (rv['fake_img'] + 1) / 2 * 255
    refined_show = (rv['r_output'] + 1) / 2 * 255

    #original_show_ = tf.minimum(((rv['fake_img_'] + 1) / 2 * 255)**2,500)/500*255

    real_iamge = tf.cast(real_show, tf.uint8)
    original = tf.cast(original_show, tf.uint8)
    refined = tf.cast(refined_show, tf.uint8)
    #original_ = tf.cast(original_show_, tf.uint8)

    if LOSS == 'simGAN':
        original_SU = tf.cast((original), tf.uint8)
        refined_SU = tf.cast((refined), tf.uint8)
    if LOSS == 'SU' or LOSS == 'SU_new':
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
    # fake_img_temp = ((fake_img + 1) / 2 * 255) ** 2
    # refined_img_temp = ((refined_img + 1) / 2 * 255) ** 2
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

        # 确保训练初始阶段稳定
        if th>250:
            th = fake_bg_sort[int(fake_bg_len*0.95)]
        # print(refined_bg_len)
        # print(th)
        # print(refined_bg_sort)
        refined_ths[image_idx] = refined_ths[image_idx] * th
        # print("== =============")

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
        s.run(tf.global_variables_initializer())  # tf.global_variables_initializer()
        writer = tf.summary.FileWriter(LOG_DIR, s.graph)
        saver = tf.train.Saver(max_to_keep=10)
        for step in range(201):
            print('------pre-------', step)
            print(LOSS)
            real_img = batch_real_new()
            fake_img, mask = batch_fake_new()

            s.run(g['r2f_train_op'],feed_dict={g['fake_img']: fake_img})
            # s.run(g['d_train_op'], feed_dict={g['real_img']: real_img, g['fake_pool']: fake_img})
            if step % 1 == 0:
                r_loss = s.run(g['r_loss'],feed_dict={g['fake_img']: fake_img})
                print('r_loss:', r_loss)
                #print('d_loss:', d_loss)
                # writer.add_summary(summary, step)
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

    image_history_buffer = ImageHistoryBuffer((0, 16, 64, 64, 1), BATCH_SIZE * 100, BATCH_SIZE)
    g = define_graph(LOSS)
    # batch_it_real = batch_real()
    # batch_it_fake = batch_fake()

    with tf.Session() as s:
        s.run(tf.global_variables_initializer())  # tf.global_variables_initializer()
        writer = tf.summary.FileWriter(LOG_DIR, s.graph)
        saver = tf.train.Saver(max_to_keep=10)

        if to_restore:
            chkpt_fname = tf.train.latest_checkpoint(MODEL_DIR)
            saver.restore(s, chkpt_fname)

        for step in range(MAX_STEPS):
            print('----------------', step)
            print(LOSS)

            real_img = batch_real_new()
            fake_img, mask = batch_fake_new()
            
            # 产生一个预先图像
            refined_image = s.run(g['r_output'],feed_dict={g['fake_img']: fake_img})
            print("before")
            print(np.sum(refined_image))
            # 初始化real_ths, fake_ths
            fake_ths, refined_ths = image_mp(fake_img, refined_image, mask)
            for step_r in range(STEP_REFINER):
                _, refined_image = s.run([g['r2f_train_op'], g['r_output']],feed_dict={g['fake_img']: fake_img, g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
                # _, fake_image_ = s.run([g['f2r_train_op'], g['fake_img_']],feed_dict={g['fake_img']: fake_img, g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
            print("after")
            print(np.sum(refined_image))
            print(np.mean(fake_ths), np.mean(refined_ths))

            for step_d in range(STEP_DISCRIMINATOR):
                # refined_image_batch = s.run(g['r_output'], feed_dict={g['real_img']: real_img, g['fake_img']: fake_img})

                if HISTORYBUFFER == 'True':
                    # use a history of refined images
                    half_batch_from_image_history = image_history_buffer.get_from_image_history_buffer()
                    image_history_buffer.add_to_image_history_buffer(refined_image)
                    if len(half_batch_from_image_history):
                        # print("yes")
                        refined_image[:BATCH_SIZE // 2] = half_batch_from_image_history

                s.run(g['d_train_op'], feed_dict={g['real_img']: real_img, g['fake_pool']: refined_image})

            if step % STEP_EVALUATE == 0:
                d_loss,r_loss,summary = s.run([g['d_loss'],g['r_loss'],g['summary']], feed_dict={g['real_img']: real_img, g['fake_img']: fake_img,g['fake_pool']: refined_image,g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
                r_loss_realism,r_loss_reg = s.run([g['r_loss_realism'],g['r_loss_reg']], feed_dict={g['fake_img']: fake_img,g['refined_ths']: refined_ths,g['fake_ths']: fake_ths, g['K']: K})
                print('d_loss:', d_loss)
                print('r_loss:', r_loss)
                print('r_loss_realism:',r_loss_realism)
                print('r_loss_reg:',r_loss_reg)
                writer.add_summary(summary, step)
            if step % 500 == 0:
                saver.save(s, os.path.join(MODEL_DIR, 'model.ckpt'), global_step=step + 1)
            print('----------------')


def test():
    with tf.Session() as s:
        MODEL_DIR = os.path.join(DATA_DIR, 'model/model_' + str(model_num))
        REFINED_DIR = os.path.join(DATA_DIR, ' ')

        if not os.path.exists(REFINED_DIR):
            os.mkdir(REFINED_DIR)

        g = define_graph(LOSS='SU_new')
        saver = tf.train.Saver()
        saver.restore(s, tf.train.latest_checkpoint(MODEL_DIR))

        test_image_num = 200
        cut_image_num = 1
        for i in range(test_image_num):
            print(i)
            max_depth = np.random.randint(4,7)
            base_length = np.random.randint(40,60)
            mean = np.random.randint(130,170)
            std = np.random.randint(15,25)
            # max_depth = np.random.randint(4,7)
            # base_length = np.random.randint(40,60)
            # mean = np.random.randint(0,20)
            # std = np.random.randint(0,10)
            
            raw, img = simulator(255, max_depth, base_length, 1, mean, std, addnoise=False, data_type=np.uint16)
            print(max_depth, base_length, mean, std)
            label = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
            # 此处为foreground的0.01
            label[raw > 10] = 1

            # 压缩为扁的图像
            img_SHAPE = img.shape
            img_new = np.zeros([int(img_SHAPE[0] // 4), int(img_SHAPE[1]), int(img_SHAPE[2])], dtype=np.uint16)
            label_new = np.zeros([int(img_SHAPE[0] // 4), int(img_SHAPE[1]), int(img_SHAPE[2])], dtype=np.uint8)
            for y in range(img_SHAPE[2]):
                img_temp = copy.deepcopy(img[:, :, y:y + 1].reshape((img_SHAPE[0], img_SHAPE[1])))
                img_temp_1 = cv.resize(img_temp, (int(img_SHAPE[0]), int(img_SHAPE[1] // 4)))
                img_temp_2 = copy.deepcopy(img_temp_1.reshape((int(img_SHAPE[0] // 4), int(img_SHAPE[1]), 1)))
                img_new[:, :, y:y + 1] = copy.deepcopy(img_temp_2)

                label_temp = copy.deepcopy(label[:, :, y:y + 1].reshape((img_SHAPE[0], img_SHAPE[1])))
                label_temp_1 = cv.resize(label_temp, (int(img_SHAPE[0]), int(img_SHAPE[1] // 4)))
                kernel = np.array(
                    [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).astype(
                    np.uint8)
                label_temp_dilate = cv.dilate(label_temp_1, kernel, iterations=1)

                label_temp_2 = copy.deepcopy(label_temp_dilate.reshape((int(img_SHAPE[0] // 4), int(img_SHAPE[1]), 1)))
                label_new[:, :, y:y + 1] = copy.deepcopy(label_temp_2)

            for z in range(img_SHAPE[0] // 4):
                label_temp = copy.deepcopy(label_new[z:z + 1, :, :].reshape((img_SHAPE[1], img_SHAPE[2])))
                kernel = np.array(
                    [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]).astype(
                    np.uint8)
                label_temp_dilate = cv.erode(label_temp, kernel, iterations=1)
                label_temp_2 = copy.deepcopy(label_temp_dilate.reshape((1, int(img_SHAPE[1]), int(img_SHAPE[2]))))
                label_new[z:z + 1, :, :] = copy.deepcopy(label_temp_2)

            image_dir = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/fake_image/' + str(i + 1) + '.tif'
            label_dir = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/label_image/' + str(i + 1) + '.gt.tif'
            refined_dir = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/refined_image/' + str(i + 1) + '.ref.tif'

            # save_tif(img_new, image_dir, np.uint16)
            # save_tif(label_new, label_dir, np.uint16)
            save_tif(img_new, image_dir, np.uint8)
            save_tif(label_new, label_dir, np.uint8)

            # img_rescale = np.sqrt(img_new) / 255 * 2 - 1
            img_rescale = img_new / 255 * 2 - 1
            d, h, w = img_rescale.shape
            steps = 8
            batch = np.zeros(dtype=np.float32, shape=[BATCH_SIZE, *SHAPE, 1])
            cnt = 0

            refined_list = []
            refined_image = np.zeros_like(img_rescale, dtype=np.float32)

            for k in range(0, d - 8, steps):
                for j in range(0, h - 16, steps * 6):
                    for i in range(0, w - 16, steps * 6):
                        patch = img_rescale[k:k + SHAPE[0], j:j + SHAPE[1], i:i + SHAPE[2]]
                        r_d, r_h, r_w = patch.shape
                        batch[cnt, :r_d, :r_h, :r_w] = patch.reshape(r_d, r_h, r_w, 1).astype(np.float32)
                        cnt += 1
                        if cnt == BATCH_SIZE:
                            refined = s.run(g['r_output'], feed_dict={g['real_img']: batch, g['fake_img']: batch,
                                                                      g['training_State']: False})
                            for ref in refined:
                                refined_list.append(ref.reshape(SHAPE))
                            cnt = 0
            if cnt > 0:
                refined = s.run(g['r_output'],
                                feed_dict={g['real_img']: batch, g['fake_img']: batch, g['training_State']: False})
                for ref in refined:
                    refined_list.append(ref.reshape(SHAPE))

            index = 0
            for k in range(0, d - 8, steps):
                for j in range(0, h - 16, steps * 6):
                    for i in range(0, w - 16, steps * 6):
                        shape = refined_image[k:k + SHAPE[0], j:j + SHAPE[1], i:i + SHAPE[2]].shape
                        ref_img = refined_list[index]
                        refined_image[k + 4:k + SHAPE[0] - 4, j + 8:j + SHAPE[1] - 8, i + 8:i + SHAPE[2] - 8] = \
                            ref_img[4:shape[0] - 4, 8:shape[1] - 8, 8:shape[2] - 8]
                        index += 1

            refined_image = ((refined_image + 1) / 2 * 255)# ** 2
            # refined_image[refined_image > 16000] = 0
            refined_image[refined_image > 254] = 0
            # save_tif(refined_image, refined_dir, np.uint16)
            save_tif(refined_image, refined_dir, np.uint8)

            size_x = 64
            size_y = 64
            channel = 16

            # cut image
            for z in range(steps, d - channel - steps + 1, steps):
                for x in range(steps * 4, h - size_x - steps * 4 + 1, steps * 4):
                    for y in range(steps * 4, w - size_y - steps * 4 + 1, steps * 4):
                        temp_refined_data = refined_image[z:z + channel, x:x + size_x, y:y + size_y]
                        temp_fake_data = img_new[z:z + channel, x:x + size_x, y:y + size_y]
                        temp_label = label_new[z:z + channel, x:x + size_x, y:y + size_y]

                        pix_value = np.sum(temp_label)

                        if pix_value >= 500:  # 50
                            newname_refined_data = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/data_divide_gan/' + str(
                                cut_image_num) + '.tif'
                            newname_fake_data = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/data_divide/' + str(
                                cut_image_num) + '.tif'
                            newname_label = '/4T/liuchao/unet/simGAN_data/GAN_image_2_new/label_divide/' + str(
                                cut_image_num) + '.gt.tif'

                            # save_tif(temp_refined_data, newname_refined_data, np.uint16)
                            # save_tif(temp_fake_data, newname_fake_data, np.uint16)
                            # save_tif(temp_label, newname_label, np.uint16)

                            save_tif(temp_refined_data, newname_refined_data, np.uint8)
                            save_tif(temp_fake_data, newname_fake_data, np.uint8)
                            save_tif(temp_label, newname_label, np.uint8)

                            cut_image_num = cut_image_num + 1
            print(cut_image_num)


if __name__ == '__main__':
    if MODE == 'train':
        if to_pre_train:
            pre_train(LOSS='simGAN')
        train(LOSS='SU_new')
    if MODE == 'test':
        test()

