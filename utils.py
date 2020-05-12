import numpy as np
import tensorflow as tf
from PIL import Image
import os
from itertools import groupby
from operator import itemgetter


def loss(logits, labels, classes_weights):
    """Loss function
    :param logits: logits
    :param labels: labels, can be multiclass
    :param classes_weights:
    """
    clss = len(logits)
    classes_weights = np.array(classes_weights)
    assert (
            classes_weights.size == clss), \
        "Number of classes ({0}) different than the dimension of loss weights ({1}).".format(
            clss, classes_weights.size)
    #
    epsilon = tf.constant(value=1e-10)
    with tf.name_scope('loss'):
        labels = tf.cast(tf.reshape(labels, (-1, clss)), tf.float32)
        cost, precision, recall, specificity, accuracy = 0., 0., 0., 0., 0.
        f1 = []
        for i in range(clss):
            lgts = tf.reshape(logits[i], (-1, 2))
            lbls = tf.one_hot(tf.cast(labels[:, i], tf.int32), depth=2)
            #
            cost += tf.math.divide(tf.reduce_mean(
                tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=lgts, labels=lbls), classes_weights[i])),
                float(clss))
            # cost_ = tf.multiply( labels * tf.log(tf.nn.softmax( lgts ) + epsilon) +
            # (1 - labels) * tf.log( 1 - tf.nn.softmax( lgts ) + epsilon), classes_weights[i])
            # cost += tf.reduce_mean( -tf.reduce_sum( cost_, axis=[1]) )
            lgts_ = tf.argmax(tf.nn.softmax(lgts), axis=1)
            lbls_ = tf.argmax(lbls, axis=1)
            #
            TP = tf.compat.v1.count_nonzero(lgts_ * lbls_, dtype=tf.float32, axis=0)
            TN = tf.compat.v1.count_nonzero((lgts_ - 1) * (lbls_ - 1), dtype=tf.float32, axis=0)
            FP = tf.compat.v1.count_nonzero(lgts_ * (lbls_ - 1), dtype=tf.float32, axis=0)
            FN = tf.compat.v1.count_nonzero((lgts_ - 1) * lbls_, dtype=tf.float32, axis=0)
            # Divide_no_NAN in case no TP exist in sample
            rec = tf.math.divide_no_nan(TP, (TP + FN))
            prec = tf.math.divide_no_nan(TP, (TP + FP))
            spec = tf.math.divide_no_nan(TN, (TN + FP))
            acc = tf.math.divide_no_nan((TP + TN), (TP + TN + FP + FN))
            # Divide by the number of classes to average metrics
            accuracy += tf.math.divide(acc, float(clss))
            recall += tf.math.divide(rec, float(clss))
            precision += tf.math.divide(prec, float(clss))
            specificity += tf.math.divide(spec, float(clss))
            # Store every F1 scores in list
            f1_ = 2 * prec * rec / (prec + rec + epsilon)
            f1 += [f1_]
            #
        f1 = tf.convert_to_tensor(f1, dtype=tf.float32)
    return cost, f1, recall, precision, specificity, accuracy


def show_progress(txt, loss, f1, rec, prec, spec, show):
    msg = "{0} --- loss: {1:.5f} --- f1: {2} --- recall: {3:.5f} --- precision: {4:.5f} --- accuracy: {5:.5f}"
    msg = msg.format(txt, loss, f1, rec, prec, spec)
    if show is True:
        print(msg)
    else:
        return msg


def remove_transparency(im, bg_colour=(255, 255, 255)):
    """Remove alpha channel if it exists
    Only process if image has transparency (http://stackoverflow.com/a/1963146)"""
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        #
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').split()[-1]
        #
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg.convert('RGB')
        #
    else:
        return im


def array_to_text(arr, n):
    """Convert f1 results into readable text"""
    txt = list(arr)  # transform array to list
    if n == 3:
        txt = ["%.3f" % item for item in txt]  # format decimals
    else:
        txt = ["%.0f" % item for item in txt]
    return ' | '.join(e for e in txt)


def get_image_array(imgs,img_size,normalize=True):
    '''
    Get array of inout image for final prediction
    with trained network
    '''
    images = []
    for i in range(len(imgs)):
        img_0 = []
        for img_ in imgs[i]:
            img = remove_transparency(Image.open(img_))
            img = img.resize( (img_size, img_size) )
            img = np.array(img)
            img = img.astype(np.float32)
            if normalize is True:
                if len(img.shape) == 3:
                    img = np.multiply(img, 1.0 / 255.0)
                else:
                    img = ( (img - np.min(img[img!=0])) / (np.max(img)-np.min(img[img!=0])) ).clip(min=0)
            img = np.reshape( img, (img_size*img_size, -1) )
            img_0.append(img)
        image = np.concatenate( img_0, axis=-1)
        image = np.reshape(image, (img_size, img_size, image.shape[-1]) )
        images.append(image)
        #
    images = np.array(images)
    return images


def get_annotation_array(imgs, img_size):
    """Get array of inout image for final prediction
    with trained network"""
    images = []
    for i in range(len(imgs)):
        img_0 = []
        for img_ in imgs[i]:
            img = remove_transparency(Image.open(img_))
            img = img.resize((img_size, img_size))
            img = img.convert('L')
            img = np.array(img.point(lambda x: 0 if x > 128 else 1), dtype='int32')
            # img = np.expand_dims(img,axis=-1)
            img_0.append(img)
        images.append(img_0)
        #
    images = np.array(images)
    images = np.moveaxis(images, 1, -1)
    #
    return images


def load_train(path):
    """Provides training from existing sets (directories)"""
    # Collect existing folders
    dir = []
    for x in os.walk(path):
        try:
            dir.append(x[0].split('\\')[1])
        except:
            pass

    # Collect png files in folders
    files_in_folders = []
    for folder in dir:
        folder_path = os.path.join(path, folder)

        # Collect data names in current folder
        files = []
        for file in os.walk(folder_path):
            files.append(file)
            files = [os.path.join(folder_path, name) for name in files[0][2]]

            # Rule out non-PNG files
            cnt = 0
            for f in files:
                formt = f.split('\\')[-1].split('.')[-1]
                if formt == 'png' or formt == 'PNG':
                    pass
                else:
                    files.pop(cnt)
                cnt += 1

        # Group images given annotation index
        ints = [int(name.split('_')[-1].split('.')[0]) for name in files]
        ints = [[idx, int_] for idx, int_ in zip(range(len(ints)), ints)]
        idxs = sorted(ints, key=itemgetter(1))
        idxs = np.array(idxs)[:, 0]
        files_ = [files[i] for i in idxs]
        files_ = [list(i) for j, i in groupby(files_, lambda a: a.split('_')[-1].split('.')[0])]
        files_in_folders.append(files_)

    return [e for sub in files_in_folders for e in sub]
