import os
import json
import yaml
import argparse
import numpy as np

import SimpleITK as sitk
import tensorflow as tf
import skimage.measure as measure

from network import vessel_attention


def get_patch(image, label, vessel):
    wh = int(image.shape[1])
    size = [30, 256, 256]
    label_l = measure.label(label)
    props = measure.regionprops(label_l)
    slices_res = []
    for p in props:
        centr_d = int(p.centroid[0])
        centr_w = int(p.centroid[1])
        centr_h = int(p.centroid[2])
        if centr_d - size[0] // 2 < 0:
            image_temp = image[0:size[0]]
            label_temp = label[0:size[0]]
            vessel_temp = vessel[0:size[0]]
        elif centr_d + size[0] // 2 > image.shape[0]:
            image_temp = image[-size[0]:]
            label_temp = label[-size[0]:]
            vessel_temp = vessel[-size[0]:]
        else:
            image_temp = image[centr_d - size[0] // 2:centr_d + size[0] // 2]
            label_temp = label[centr_d - size[0] // 2:centr_d + size[0] // 2]
            vessel_temp = vessel[centr_d - size[0] // 2:centr_d + size[0] // 2]

        if centr_w - size[1] // 2 < 0:
            image_temp = image_temp[:, 0:size[1]]
            label_temp = label_temp[:, 0:size[1]]
            vessel_temp = vessel_temp[:, 0:size[1]]
        elif centr_w + size[1] // 2 > wh:
            image_temp = image_temp[:, -size[1]:]
            label_temp = label_temp[:, -size[1]:]
            vessel_temp = vessel_temp[:, -size[1]:]
        else:
            image_temp = image_temp[:, centr_w - size[1] // 2:centr_w + size[1] // 2]
            label_temp = label_temp[:, centr_w - size[1] // 2:centr_w + size[1] // 2]
            vessel_temp = vessel_temp[:, centr_w - size[1] // 2:centr_w + size[1] // 2]

        if centr_h - size[2] // 2 < 0:
            image_temp = image_temp[..., 0:size[2]]
            label_temp = label_temp[..., 0:size[2]]
            vessel_temp = vessel_temp[..., 0:size[2]]
        elif centr_h + size[2] // 2 > wh:
            image_temp = image_temp[..., -size[2]:]
            label_temp = label_temp[..., -size[2]:]
            vessel_temp = vessel_temp[..., -size[2]:]
        else:
            image_temp = image_temp[..., centr_h - size[2] // 2:centr_h + size[2] // 2]
            label_temp = label_temp[..., centr_h - size[2] // 2:centr_h + size[2] // 2]
            vessel_temp = vessel_temp[..., centr_h - size[2] // 2:centr_h + size[2] // 2]

        slices_res.append([np.expand_dims(image_temp, axis=-1),
                           np.expand_dims(label_temp, axis=-1),
                           np.expand_dims(vessel_temp, axis=-1)])

    return slices_res


def _process_data(data, a_min=-1024, a_max=2048):
    data = np.clip(data, a_min, a_max)
    data -= a_min
    data = data / (a_max - a_min)
    return data


def data_generator(file_list, batch_size=1, is_training=True):
    data_list = []
    for i in range(len(file_list)):
        img_path = os.path.join(file_list[i], 'image.nii.gz')
        label_path = os.path.join(file_list[i], 'aneurysm_mask.nii.gz')
        vessel_path = os.path.join(file_list[i], 'vessel_mask.nii.gz')
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue
        img_itk = sitk.ReadImage(img_path)
        image = sitk.GetArrayFromImage(img_itk)
        image = _process_data(image, a_min=-1024, a_max=2048)
        vol_per_voxel = img_itk.GetSpacing()[0] * img_itk.GetSpacing()[1] * img_itk.GetSpacing()[2]
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        vessel = sitk.GetArrayFromImage(sitk.ReadImage(vessel_path))

        label[label != 0] = 1

        slices_list = get_patch(image, label, vessel)
        data_list += slices_list

    X = np.array([it[0] for it in data_list])
    Y = np.array([it[1] for it in data_list])
    V = np.array([it[2] for it in data_list])
    while True:
        if is_training:
            np.random.seed(123)
            np.random.shuffle(X)
            np.random.seed(123)
            np.random.shuffle(Y)
            np.random.seed(123)
            np.random.shuffle(V)

        for step in range(X.shape[0] // batch_size):
            x = X[step * batch_size:(step + 1) * batch_size, ...]
            y = Y[step * batch_size:(step + 1) * batch_size, ...]
            v = V[step * batch_size:(step + 1) * batch_size, ...]
            yield x, y, v


def explogTVSK_loss(gt, pred, weight=240, \
                    w_d=.8, w_c=.2, gm_d=.3, gm_c=.3, alpha=.1, beta=.9, \
                    eps=1., ):
    """{}"""

    gt = tf.cast(gt, dtype=tf.float32)

    p0 = tf.reshape(pred, (-1, 2))[..., 0]
    g0 = tf.reshape(gt, (-1, 1))[..., 0]
    g1 = tf.ones_like(g0) - g0
    p1 = tf.ones_like(p0) - p0
    dice = (tf.reduce_sum(p0 * g0) + eps) / \
           (tf.reduce_sum(p0 * g0) + alpha * tf.reduce_sum(p0 * g1) + beta * tf.reduce_sum(p1 * g0) + eps)
    prob = g1 * p1 + g0 * p0
    prob_smooth = tf.clip_by_value(prob, 1e-15, 1.0 - 1e-7)

    LD = tf.reduce_sum(tf.pow(-tf.log(dice), gm_d))
    LC = tf.reduce_sum(tf.pow(-tf.log(prob_smooth), gm_c) * weight) \
         / tf.cast(tf.shape(prob_smooth)[0], dtype=tf.float32)

    return LD * w_d + LC * w_c


def pickrandom_slice_ingraph_with_Vas(image, label, vessel, img_size, depth=8, size=384):
    '''
    {PICK MID SLICES, *image and label at same time}
    Input: img[tf.Tensor] -> [depth, x-axis, y-axis]
           label[tf.Tensor] -> [depth, x-axis, y-axis]
           img_size[list] -> Size of 3-d image
           depth[int] -> target depth
           size[int] -> target x,y axis
    Output: img_out -> [target_depth, x-axis, y-axis]
    '''
    condi = tf.Assert(tf.greater_equal(img_size[0], depth),
                      ['Image depth smaller than target depth'])
    with tf.control_dependencies([condi]):
        st_idx_d = tf.random_uniform(
            [1], maxval=img_size[0] - depth + 1, dtype=tf.int32)[0]
        st_idx_x = tf.random_uniform(
            [1], maxval=img_size[1] - size + 1, dtype=tf.int32)[0]
        st_idx_y = tf.random_uniform(
            [1], maxval=img_size[1] - size + 1, dtype=tf.int32)[0]
        img_out = image[:, st_idx_d:st_idx_d + depth, \
                  st_idx_x:st_idx_x + size, st_idx_y:st_idx_y + size]
        label_out = label[:, st_idx_d:st_idx_d + depth, \
                    st_idx_x:st_idx_x + size, st_idx_y:st_idx_y + size]
        vessel_out = vessel[:, st_idx_d:st_idx_d + depth, \
                     st_idx_x:st_idx_x + size, st_idx_y:st_idx_y + size]

    return img_out, label_out, vessel_out


def train(train_file, valid_file, epochs, batch=4, warm_up=True, display_step=20, reload_path='',
          ckpt_dir='./ckpt_output'):
    image_size = [30, 256, 256]
    input_size = [24, 256, 256]
    is_training = tf.placeholder(tf.bool)
    input_image = tf.placeholder(dtype=tf.float32, shape=[batch, image_size[0], image_size[1], image_size[2], 1])
    input_label = tf.placeholder(dtype=tf.float32, shape=[batch, image_size[0], image_size[1], image_size[2], 1])
    input_vessel = tf.placeholder(dtype=tf.float32, shape=[batch, image_size[0], image_size[1], image_size[2], 1])

    input_image_new, input_label_new, input_vessel_new = pickrandom_slice_ingraph_with_Vas(input_image, input_label,
                                                                                           input_vessel,
                                                                                           img_size=image_size,
                                                                                           depth=input_size[0],
                                                                                           size=input_size[1])
    input_image_new = tf.reshape(input_image_new, shape=[batch, input_size[0], input_size[1], input_size[2], 1])

    logit_label, logit_vessel = vessel_attention(input_image_new, base_channel=16, is_training=is_training,
                                                 reuse=tf.AUTO_REUSE)

    if warm_up:
        seg_loss = explogTVSK_loss(input_label_new, logit_label, weight=60)
        vsl_loss = explogTVSK_loss(input_vessel_new, logit_vessel, weight=60)
    else:
        seg_loss = explogTVSK_loss(input_label_new, logit_label)
        vsl_loss = explogTVSK_loss(input_vessel_new, logit_vessel)
    seg_loss_train = seg_loss + 0.1 * vsl_loss

    global_step = 0
    if reload_path != '':
        global_step = int(reload_path.split('-')[-1])
    global_step = tf.Variable(global_step, name='global_step', trainable=False)

    lr_decay = tf.train.cosine_decay(1e-4, global_step, 30000, alpha=0.2)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_decay)
    train_opt = optimizer.minimize(seg_loss_train, global_step=global_step)

    train_dataset = data_generator(train_file, batch_size=batch, is_training=True)
    step_train = len(train_file)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=epochs)

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        if reload_path != '':
            saver.restore(sess, reload_path)
        else:
            sess.run(init_op)

        for epoch in range(epochs):
            total_loss = []
            for step in range(step_train):
                x, y, v = next(train_dataset)
                _, loss = sess.run([train_opt, seg_loss_train], feed_dict={input_image: x,
                                                                           input_label: y,
                                                                           input_vessel: v,
                                                                           is_training: True})
                total_loss.append(loss)
                if step % display_step == 0:
                    print('*' * 20, 'Epoch {:}, train steps {:}, loss={:.4f}'.format(epoch, step, loss), flush=True)
            print('*' * 20, 'Epoch {:}, train Avg loss={:.4f}'.format(epoch, np.mean(total_loss)))

            ckpt_path = saver.save(sess, os.path.join(ckpt_dir, 'segan_net.ckpt'), global_step=global_step)
    return ckpt_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training")
    parser.add_argument("--config", type=str, default="configs/aneurysm_seg.yaml", help="Configuration file to use")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    with open(cfg['train']['train_file'], 'r') as fp:
        train_file = json.load(fp)
    with open(cfg['train']['valid_file'], 'r') as fp:
        valid_file = json.load(fp)

    model_path = train(train_file, valid_file, epochs=cfg['train']['epcohs'],
                       warm_up=True, batch=cfg['train']['batch_size'])
    tf.reset_default_graph()
    train(train_file, valid_file, epochs=cfg['train']['epochs'], warm_up=False,
          reload_path=model_path, batch=cfg['train']['batch_size'])
