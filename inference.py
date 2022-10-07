import os
import argparse
import yaml
import json
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
from skimage.morphology import remove_small_objects as remove_small_objects

from network import vessel_attention


def filter_small_cntcpt_v2(label, thresh=180, ):
    '''
    {}
    '''
    return remove_small_objects(label.astype(bool), min_size=thresh, connectivity=2)


def CTA_norm(image, a_min=-1024, a_max=2048):
    data = np.clip(image, a_min, a_max)
    data -= a_min
    data = data / (a_max - a_min)
    return data


def _thresh_perc(img, perc=97):
    '''
    {set perc percent small pixels to 0}
    Input:
        img[np.array]:
        perc[float]: percentage with range of 1-100
    Return:
        res[np.array]: image after treshold
    '''
    res = img.copy()
    p = np.percentile(res, perc)
    res[res < p] = 0
    res[res != 0] = 1
    return res


def get_crop_wvas(image, half_size=128):
    headmask = _thresh_perc(image)
    idx_tuple = np.where(headmask == 1)

    dmin, dmax = idx_tuple[0].min(), idx_tuple[0].max()
    wmin, wmax = idx_tuple[1].min(), idx_tuple[1].max()
    hmin, hmax = idx_tuple[2].min(), idx_tuple[2].max()

    w_shift = 0

    mid_d = (dmax - dmin) // 2 + 1 + dmin
    mid_w = (wmax - wmin) // 2 + 1 + wmin + w_shift
    mid_h = (hmax - hmin) // 2 + 1 + hmin

    image_crop = image[:, \
                 mid_w - half_size:mid_w + half_size, \
                 mid_h - half_size:mid_h + half_size]
    addi_data = [image.shape, mid_w, mid_h, half_size]
    return image_crop, addi_data


def re_crop_723(label, addi_data):
    org_shape = addi_data[0]
    mid_w = addi_data[1]
    mid_h = addi_data[2]
    half_size = addi_data[3]
    pad = np.zeros(org_shape)
    pad[:, \
    mid_w - half_size:mid_w + half_size, \
    mid_h - half_size:mid_h + half_size, ] = label
    return pad


def slice_wvas(ov=16, sl_depth=24):
    def _slice_wvas(image, ov=ov, depth=None, sl_depth=sl_depth):
        depth = image.shape[0]
        slc_num = (depth - sl_depth) // ov + 1
        start_idx = ((depth + sl_depth) % ov) // 2
        slices = []
        #         slices_vas = []
        idxes = []
        for i in range(slc_num):
            slices.append(image[start_idx + i * ov:start_idx + i * ov + sl_depth])
            idxes.append([start_idx + i * ov])
        _ = None
        return slices, _

    return _slice_wvas


def re_slice_723(ov=16, sl_depth=24):
    def _re_slice_723(slices, _, shape, ov=ov, sl_depth=sl_depth):
        depth = shape[0]
        slc_num = (depth - sl_depth) // ov + 1
        start_idx = ((depth + sl_depth) % ov) // 2
        img = np.zeros(shape)
        lb = (sl_depth - ov) // 2  # low boundary
        hb = (sl_depth - ov) // 2 + ov  # high boundary
        for i in range(slc_num):
            if i == 0:
                img[start_idx + i * ov + 0 + 1:start_idx + i * ov + hb] \
                    = slices[i][1:hb]
            elif i == (slc_num - 1):
                img[start_idx + i * ov + lb:start_idx + i * ov + sl_depth] \
                    = slices[i][lb:sl_depth]
            else:
                img[start_idx + i * ov + lb:start_idx + i * ov + hb] \
                    = slices[i][lb:hb]
        return img

    return _re_slice_723


###################################class###################################

batch_size = 1
slice_depth = 24
overlap_index = 16
input_size = [24, 256, 256]


class aneurysm_model():
    def __init__(self, model_path):
        self.model_path = model_path
        self.proc_thr = .75
        self.drop_objs_thr = 400
        self.creat_tf_client()

    def creat_tf_client(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_sess = tf.Session(config=config)

        self.tf_inputs = tf.placeholder(dtype=tf.float32,
                                        shape=[batch_size, input_size[0], input_size[1], input_size[2], 1])
        self.tf_outputs, _ = vessel_attention(self.tf_inputs, base_channel=16, is_training=False)

        meta = self.model_path + '.meta'
        saver = tf.train.Saver()
        saver.restore(self.tf_sess, self.model_path)

    def preprocessing(self, image):
        image_croped, addi_data = get_crop_wvas(image)
        image_croped = CTA_norm(image_croped)
        slices, slice_idxes = slice_wvas(ov=overlap_index, sl_depth=slice_depth)(image_croped)
        return slices, addi_data, slice_idxes, image_croped.shape

    def postporcessing(self, pred_patches, addi_data, slice_idxes, croped_shape, vsl_seg):
        # reconstract
        pred_mask = re_slice_723(ov=overlap_index, sl_depth=slice_depth)(pred_patches, slice_idxes, croped_shape)
        pred_mask = re_crop_723(pred_mask, addi_data)

        thresh = .8
        thresh_s_ob = 10
        #
        pred_mask[pred_mask > thresh] = 1
        pred_mask[pred_mask < 1] = 0

        pred_pure = filter_small_cntcpt_v2(pred_mask, thresh=thresh_s_ob)

        voxel_thresh = 150
        pred_pure = filter_small_cntcpt_v2(pred_mask, thresh=voxel_thresh)

        pred_pure *= vsl_seg.astype(bool)

        return pred_pure

    def inference(self, image, vessel_seg):
        patches, addi_data, slice_idxes, image_cropped_shape = self.preprocessing(image)

        pred_patches = []
        for i, patch in enumerate(patches):
            patch_in = np.expand_dims(patch, axis=0)
            patch_in = np.expand_dims(patch_in, axis=-1)

            sess_out = self.tf_sess.run(
                self.tf_outputs,
                feed_dict={self.tf_inputs: patch_in}
            )
            pred_patches.append(sess_out[0, ..., 1])

        # reconstract patch
        pred_mask = self.postporcessing(
            pred_patches, addi_data, slice_idxes, image_cropped_shape, vessel_seg)

        return pred_mask


def load_data(file_path):
    img_path = os.path.join(file_path, 'image.nii.gz')
    vessel_path = os.path.join(file_path, 'vessel_mask.nii.gz')
    img_itk = sitk.ReadImage(img_path)
    image = sitk.GetArrayFromImage(img_itk)
    vessel = sitk.GetArrayFromImage(sitk.ReadImage(vessel_path))
    return image, vessel, [img_itk.GetSpacing(), img_itk.GetOrigin()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="inference")
    parser.add_argument("--config", type=str, default="configs/aneurysm_seg.yaml", help="Configuration file to use")
    parser.add_argument("--ckpt", type=str, default="ckpt_output/segan_net.ckpt-24", help="Checkpoint file to use")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    with open(cfg['inference']['test_file'], 'r') as fp:
        test_file = json.load(fp)

    infr = aneurysm_model(args.ckpt)
    for path in test_file:
        image, vessel, conf = load_data(path)
        pred = infr.inference(image, vessel)

        itk_img = sitk.GetImageFromArray(pred.astype(np.uint8))
        itk_img.SetSpacing(conf[0])
        itk_img.SetOrigin(conf[1])
        sitk.WriteImage(itk_img, os.path.join(path, 'aneurysm_pred.nii.gz'))
