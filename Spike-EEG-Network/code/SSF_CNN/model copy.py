#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   model.py

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2021/9/15 1:13   lintean      1.0         None
'''

import math
import time
import random
import numpy as np
import pandas as pd
from dotmap import DotMap
from utils import cart2sph, pol2cart, makePath
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
# from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
# from keras.models import Sequential
# import keras.backend as K
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
from keras.utils import np_utils
from scipy.io import loadmat
# import keras
import os
from importlib import reload
from tqdm import tqdm
np.set_printoptions(suppress=True)


def get_logger(name, log_path):
    import logging
    reload(logging)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = makePath(log_path) + "/Train_" + name + ".log"
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    if log_path == "./result/test":
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, args):
    locs = loadmat('locs_orig.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):args.image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):args.image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images


def read_prepared_data(args):
    data = []

    for l in range(len(args.ConType)):
        for k in range(args.trail_number):
            filename = args.data_document_path + "/" + args.ConType[l] + "/" + args.name + "Tra" + str(k + 1) + ".csv"
            data_pf = pd.read_csv(filename, header=None)
            eeg_data = data_pf.iloc[:, 2 * args.audio_channel:]

            data.append(eeg_data)

    data = pd.concat(data, axis=0, ignore_index=True)
    return data


# output shape: [(time, feature) (window, feature) (window, feature)]
def window_split(data, args):
    random.seed(args.random_seed)
    # init
    test_percent = args.test_percent
    window_lap = args.window_length * (1 - args.overlap)
    overlap_distance = max(0, math.floor(1 / (1 - args.overlap)) - 1)

    train_set = []
    test_set = []

    for l in range(len(args.ConType)):
        label = pd.read_csv(args.data_document_path + "/csv/" + args.name + args.ConType[l] + ".csv")

        # split trial
        for k in range(args.trail_number):
            # the number of windows in a trial
            window_number = math.floor(
                (args.cell_number - args.window_length) / window_lap) + 1

            test_window_length = math.floor(
                (args.cell_number * test_percent - args.window_length) / window_lap)
            test_window_length = test_window_length if test_percent == 0 else max(
                0, test_window_length)
            test_window_length = test_window_length + 1

            test_window_left = random.randint(0, window_number - test_window_length)
            test_window_right = test_window_left + test_window_length - 1
            target = label.iloc[k, args.label_col]

            # split window
            for i in range(window_number):
                left = math.floor(k * args.cell_number + i * window_lap)
                right = math.floor(left + args.window_length)
                # train set or test set
                if test_window_left > test_window_right or test_window_left - i > overlap_distance or i - test_window_right > overlap_distance:
                    train_set.append(np.array([left, right, target, len(train_set), k, args.subject_number]))
                elif test_window_left <= i <= test_window_right:
                    test_set.append(np.array([left, right, target, len(test_set), k, args.subject_number]))

    # concat
    train_set = np.stack(train_set, axis=0)
    test_set = np.stack(test_set, axis=0) if len(test_set) > 1 else None

    return np.array(data), train_set, test_set



complex_i=0+1j
## 输入到RF神经元中进行处理
## 共振神经元
class RF():
    def __init__(self,w,delta_t,lambda_):
        self.w=w
        self.z=0*complex_i
        self.delta_t=delta_t
        self.lambda_=lambda_
    
    def resonate(self,a):
        self.z = self.lambda_ * np.exp(complex_i * self.w * self.delta_t) * self.z+ a

## 进行近似的fft的变换
## 本质上FFT的衰减窗口是整个窗口，对应的结果是完全相同的
winlen=128

def RF_FFT(x,N,delta_t=1,lamb=0.80):
    w=-2 * np.pi/N
    population=[RF(w*i,delta_t,lamb) for i in range(N)]
    result=np.zeros((x.shape[0],N,x.shape[1]),dtype=np.complex128)
    for i in range(x.shape[0]):
        ## 生成复数列形式的temp
        temp=np.zeros((N,x.shape[1]))*complex_i
        for j in range(N):
            population[j].resonate(x[i])
        for j in range(N):
             temp[j]=population[j].z
        result[i]=temp.copy()
    
    ## 为了保证相应的值符合相应的条件：
    windows=winlen//2
    winstep=windows//2
    winnum=(winlen-windows)//winstep+1
    answer=np.zeros(result[0].shape,dtype=np.complex128)
    for i in range(winnum):
        answer+=result[i]
    ## 如何只保留最后一个 实际上batchsize==64
    ## 这样返回的是相应的fft结果 result[:,:,-1]
    ## 先暂时利用fft的结果进行计算 看是否于原本的结果相同
    # print(answer.shape) ## 输出相应的大小
    # return result[-1]
    return answer

def to_alpha(data, window, args):
    alpha_data = []
    for window_index in tqdm(range(window.shape[0])):
        start = window[window_index][args.window_metadata.start]
        end = window[window_index][args.window_metadata.end]
        rf_window_data = np.fft.fft(data[start:end, :], n=args.window_length, axis=0)
        # rf_window_data=RF_FFT(data[start:end, :], N=args.window_length)
        rf_window_data = np.abs(rf_window_data) / args.window_length
        rf_window_data = np.sum(np.power(rf_window_data[args.point_low:args.point_high, :], 2), axis=0)
        alpha_data.append(rf_window_data)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data




def main(name="S1", data_document_path="E:/KUL_single_single_snn_1to32_mean"):
    args = DotMap()
    args.name = name
    args.subject_number = int(args.name[1:])
    args.data_document_path = data_document_path
    args.ConType = ["No"]
    args.fs = 128
    args.window_length = math.ceil(args.fs * 5)
    args.overlap = 0.5
    args.batch_size = 32
    args.max_epoch = 200
    args.random_seed = time.time()
    args.image_size = 32
    args.people_number = 16
    args.eeg_channel = 64
    args.audio_channel = 1
    args.channel_number = args.eeg_channel + args.audio_channel * 2
    args.trail_number = 8
    args.cell_number = 46080
    args.test_percent = 0.1
    args.vali_percent = 0.1
    args.label_col = 0
    args.alpha_low = 8
    args.alpha_high = 13
    args.log_path = "./result"
    args.frequency_resolution = args.fs / args.window_length
    args.point_low = math.ceil(args.alpha_low / args.frequency_resolution)
    args.point_high = math.ceil(args.alpha_high / args.frequency_resolution) + 1
    args.window_metadata = DotMap(start=0, end=1, target=2, index=3, trail_number=4, subject_number=5)
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"
    logger = get_logger(args.name, args.log_path)
    print("-----------{}\{}--------".format(name,args.window_length))
    # load data 和 label
    data = read_prepared_data(args)

    # split window、testset
    data, train_window, test_window = window_split(data, args)
    train_label = train_window[:, args.window_metadata.target]
    test_label = test_window[:, args.window_metadata.target]

    # fft
    train_data = to_alpha(data, train_window, args)
    test_data = to_alpha(data, test_window, args)
    del data

    # to images
    train_data = gen_images(train_data, args)
    test_data = gen_images(test_data, args)

    train_data = np.expand_dims(train_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)
    train_label = np_utils.to_categorical(train_label - 1, 2)
    test_label = np_utils.to_categorical(test_label - 1, 2)

    np.savez(f"{makePath(f'E:/KUL_single_single_snn_1to32_mean/post/{args.window_length}s')}/{name}",
             train_images=train_data,test_images=test_data,
            train_label=train_label, test_label=test_label)
    # # train
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), padding='same', input_shape=train_data.shape[1:],
    #                  kernel_regularizer=keras.regularizers.l2(0.01), data_format="channels_last"))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_last"))
    # model.add(Dropout(0.1))

    # model.add(Flatten())

    # model.add(Dense(512))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dropout(0.3))

    # model.add(Dense(32))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))

    # # Output layer
    # model.add(Dense(2))
    # model.add(Activation('softmax'))

    # # Output the parameter status of each layer of the model
    # model.summary()

    # opt = keras.optimizers.RMSprop(lr=0.0003, decay=3e-4)
    # model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    # # plot_model(model, to_file='model.png', show_shapes=True)

    # history = model.fit(train_data,    train_label, batch_size=args.batch_size, epochs=args.max_epoch, validation_split=args.vali_percent, verbose=2)
    # loss, accuracy = model.evaluate(test_data, test_label)
    # print(loss, accuracy)
    # logger.info(loss)
    # logger.info(accuracy)


if __name__ == "__main__":
    # main(name="S5")
    for i in range(1,17): main("S"+str(i))

