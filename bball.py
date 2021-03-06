#!/usr/bin/env python

import numpy as np
from keras.layers import Convolution2D, Activation, merge, Deconvolution2D, \
        Input, Lambda, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.models import Model
import keras.backend as K
from vidextend.flow import flow_from_directory


img_shape = (224, 224)
start_img = 10
num_in_images = 10
num_out_images = 10
batch_size = 1
datadir = '/usr/share/vid/content/content/'
# datadir = '/home/ubuntu/courses/deeplearning2/data/bbal'


def get_flows(ddir, x, y):
    return flow_from_directory(datadir, x[0], x[1], y[1], target_size=img_shape)


def conv_block(x, filters, size, stride=(2, 2), mode='same'):
    x = Convolution2D(filters, size, size, subsample=stride, border_mode=mode)(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x)


def res_block(ip, nf=64):
    x = conv_block(ip, nf, 3, (1, 1))
    x = Convolution2D(nf, 3, 3, border_mode='same')(x)
    x = BatchNormalization(mode=2)(x)
    return merge([x, ip], mode='sum')


def deconv_block(x, filters, size, shape, stride=(2, 2)):
    x = Deconvolution2D(filters, size, size, subsample=stride, border_mode='same',
                        output_shape=(None,)+shape)(x)
    x = BatchNormalization(mode=2)(x)
    return Activation('relu')(x)


def preproc(x):
    rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    return (x - rn_mean)[:, :, :, ::-1]


def get_network(num_in, num_out):
    # Some preamble

    source_tensor_shape = (img_shape[0], num_in*img_shape[1], 3)
    dest_tensor_shape = (img_shape[0], num_out*img_shape[1], 3)

    inp = Input(source_tensor_shape)

    x = conv_block(inp, 64, 16, (1, 1))
    x = AveragePooling2D((4, 4))(x)

    x = conv_block(x, 64, 9, (1, 1))
    x = AveragePooling2D((4, 4))(x)

    for i in range(4):
        x = res_block(x)
    x = deconv_block(x, 64, 3, (dest_tensor_shape[0] // 8, dest_tensor_shape[1] // 8, 64))
    x = deconv_block(x, 64, 3, (dest_tensor_shape[0] // 4, dest_tensor_shape[1] // 4, 64))
    x = deconv_block(x, 64, 3, (dest_tensor_shape[0], dest_tensor_shape[1], 64), stride=(4, 4))
    # x = deconv_block(x, 64, 3, (dest_tensor_shape[0] * 2 * 2, dest_tensor_shape[1] * 2 * 2, 64))
    x = Convolution2D(3, 9, 9, activation='tanh', border_mode='same', subsample=(1, 1))(x)
    outp = Lambda(lambda x: (x+1)*127.5)(x)

    M = Model(inp, outp)
    M.summary()

    vgg_l = Lambda(preproc)
    outp_l = vgg_l(outp)

    vgg_inp = Input(dest_tensor_shape)
    vgg = VGG16(include_top=False, input_tensor=vgg_l(vgg_inp))
    for l in vgg.layers:
        l.trainable = False

    vgg_content = Model(vgg_inp, vgg.get_layer('block2_conv2').output)
    vgg1 = vgg_content(vgg_inp)
    vgg2 = vgg_content(outp_l)

    loss = Lambda(lambda x: K.sqrt(K.mean((x[0]-x[1])**2, (1, 2))))([vgg1, vgg2])
    m_final = Model([inp, vgg_inp], loss)

    m_final.compile('adam', 'mse')

    return m_final, inp, outp


def main():
    xyflow = get_flows(datadir, (start_img, start_img + num_in_images - 1),
                                (start_img + num_in_images,
                                 start_img + num_in_images + num_out_images - 1))

    m, inp, outp = get_network(num_in_images, num_out_images)
    targ = np.zeros((batch_size, 128))

    for (x, y) in xyflow:
        m.fit([x.reshape((1, ) + x.shape),
               y.reshape((1, ) + y.shape)], targ, 1, 20)

    if False:
        # K.set_value(m_final.optimizer.lr, 1e-4)
        # m_final.fit([xflow, yflow], targ, 16, 2)

        print("saving weights ...")
        m.save_weights('m_final_full.h5')
        top_model = Model(inp, outp)
        top_model.save_weights('top_final.h5')
    print("Done")


if __name__ == '__main__':
    main()
