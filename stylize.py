# Copyright (c) 2015-2019 Anish Athalye. Released under GPLv3.

import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from functools import reduce
import vgg

CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')


def stylize(network, content, styles, iterations,
            content_weight, content_weight_blend, style_weight, style_layer_weight_exp, style_blend_weights, tv_weight,
            learning_rate, beta1, beta2, epsilon, pooling):
    """
    Stylize images.

    This function yields tuples (iteration, image, loss_vals) at every
    iteration. However `image` and `loss_vals` are None by default. Each, `image` is not None. Each ,
    `loss_vals` is not None.
    `loss_vals` is a dict with loss values for the current iteration, e.g.
    ``{'content': 1.23, 'style': 4.56, 'tv': 7.89, 'total': 13.68}``.
    :rtype: iterator[tuple[int,image]]
    """
    shape = (1,) + content.shape  # 若content.shape=(356, 600, 3)  shape=(356, 600, 3, 1)
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}  # 创建内容features map
    style_features = [{} for _ in styles]  # 创建风格features map

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)  # 加载预训练模型，得到weights和mean_pixel

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp
        # style_layer_weight_exp默认为1不增长

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum  # 更新style_layers_weights，使其总和为1

    # 首先创建一个image的占位符，然后通过eval()的feed_dict将content_pre传给image，
    # 启动net的运算过程，得到了content的feature maps
    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:  # 计算content features
        image = tf.placeholder('float', shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)  # 所有网络在此构建，net为content的features maps
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])  # content - vgg_mean_pixel
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})  # content_features取值

    # compute style features in feedforward mode
    for i in range(len(styles)):  # 计算style features
        g = tf.Graph()
        with g.as_default(), g.device('/cpu:0'), tf.Session() as sess:
            image = tf.placeholder('float', shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)  # pooling 默认为MAX
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])  # styles[i]-vgg_mean_pixel
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                features = np.reshape(features, (-1, features.shape[3]))  # 根据通道数目reshape
                gram = np.matmul(features.T, features) / features.size  # gram矩阵
                style_features[i][layer] = gram

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(shape) * 0.256  # 初始化图片
        image = tf.Variable(initial)
        '''
                image = tf.Variable(initial)初始化了一个TensorFlow的变量，即为我们需要训练的对象。
                注意这里我们训练的对象是一张图像，而不是weight和bias。
                '''
        net = vgg.net_preloaded(vgg_weights, image, pooling)  # 此处的net为生成图片的features map

        # content loss
        content_layers_weights = {}
        content_layers_weights[
            'relu4_2'] = content_weight_blend  # 内容图片 content weight blend, conv4_2 * blend + conv5_2 * (1-blend)
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend  # content weight blend默认为1，即只用conv4_2层

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(content_layers_weights[content_layer] * content_weight * (2 * tf.nn.l2_loss(#生成图片-内容图片
                net[content_layer] - content_features[content_layer]) / content_features[content_layer].size))# tf.nn.l2_loss：output = sum(t ** 2) / 2
        content_loss += reduce(tf.add, content_losses)

        # style loss
        style_loss = 0
        '''
                由于style图像可以输入多幅，这里使用for循环。同样的，将style_pre传给image占位符，
                启动net运算，得到了style的feature maps，由于style为不同filter响应的内积，
                因此在这里增加了一步：gram = np.matmul(features.T, features) / features.size，即为style的feature。
                '''
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size  #求得生成图片的gram矩阵
                style_gram = style_features[i][style_layer]
                style_losses.append(
                    style_layers_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size)
            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = tv_weight * 2 * (
            (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :shape[1] - 1, :, :]) /
             tv_y_size) +
            (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :shape[2] - 1, :]) /
             tv_x_size))

        # total loss
        '''
                接下来定义了Content Loss和Style Loss，结合文中的公式很容易看懂，在代码中，
                还增加了total variation denoising，因此总的loss = content_loss + style_loss + tv_loss
                '''
        loss = content_loss + style_loss + tv_loss  #总loss为三个loss之和

        # We use OrderedDict to make sure we have the same order of loss types
        # (content, tv, style, total) as defined by the initial costruction of
        # the loss_store dict. This is important for print_progress() and
        # saving loss_arrs (column order) in the main script.
        #
        # Subtle Gotcha (tested with Python 3.5): The syntax
        # OrderedDict(key1=val1, key2=val2, ...) does /not/ create the same
        # order since, apparently, it first creates a normal dict with random
        # order (< Python 3.7) and then wraps that in an OrderedDict. We have
        # to pass in a data structure which is already ordered. I'd call this a
        # bug, since both constructor syntax variants result in different
        # objects. In 3.6, the order is preserved in dict() in CPython, in 3.7
        # they finally made it part of the language spec. Thank you!
        loss_store = OrderedDict([('content', content_loss),
                                  ('style', style_loss),
                                  ('tv', tv_loss),
                                  ('total', loss)])

        # optimizer setup
        # optimizer setup
        # 创建train_step，使用Adam优化器，优化对象是上面的loss
        # 优化过程，通过迭代使用train_step来最小化loss，最终得到一个best，即为训练优化的结果
        train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        # optimization
        best_loss = float('inf')
        best = None
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Optimization started...')
            for i in range(iterations):
                print('Iteration %4d/%4d' % (i + 1, iterations))
                train_step.run()

                last_step = (i == iterations - 1)
                if last_step:
                    loss_vals = get_loss_vals(loss_store)
                    print_progress(loss_vals)
                else:
                    loss_vals = None

                if last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)
                else:
                    img_out = None

                yield img_out


def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def get_loss_vals(loss_store):
    return OrderedDict((key, val.eval()) for key, val in loss_store.items())


def print_progress(loss_vals):
    for key, val in loss_vals.items():
        print('{:>13s} {:g}'.format(key + ' loss:', val))
