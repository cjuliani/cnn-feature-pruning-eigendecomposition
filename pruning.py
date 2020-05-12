from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

import cvmodel
import math
import os
import utils

tf.disable_v2_behavior()


class classifier:
    def __init__(self):
        self.img_path = './data/images'
        self.anno_path = './data/annotations'
        self.ft_path = './feature_maps/'
        self.model_path = './checkpoint/'
        self.model_name = 'segmentation.ckpt-285'
        self.model = os.path.join(self.model_path, self.model_name)

        # Parameters
        self.depth = 7
        self.classes = 1
        self.img_size = 32

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, None, None, self.depth], name='input')
        self.y_true = tf.placeholder(tf.float32, shape=[None, None, None, self.classes], name='y_true')
        self.rate = tf.placeholder(tf.float32, name='dropout_rate')
        self.is_training = tf.placeholder(tf.bool, shape=())

        # Build network
        self.y01 = cvmodel.build_model(input=self.x,
                                       drop_rate=0,
                                       is_training=False)

        # Calculate loss + f1
        self.cost_reg, self.f1_vec, self.recall, \
        self.precision, self.specificity, self.accuracy = utils.loss(logits=[self.y01],
                                                      labels=self.y_true,
                                                      classes_weights=[2.])
        # Open session and restore model
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.model)

        # Load data
        self.img_names = utils.load_train(path=self.img_path)
        self.anno_names = utils.load_train(path=self.anno_path)
        self.imgs_ = utils.get_image_array(self.img_names, self.img_size)
        self.annos_ = utils.get_annotation_array(self.anno_names, self.img_size)
        n = self.imgs_.shape[0]

        print('\nNumber of images:', n)
        # Get number of trainable variables
        v_nb = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print('Number of trainable variables:', v_nb)

    def predict(self, N, intv, show_avg=True, show_pgr=True):
        """Get prediction metrics to evaluate how pruning influences the model performance
        :param N: number of inputs
        :param intv: display progression at given interval
        :param show_avg: display average metrics
        :param show_pgr: display step metrics"""
        avg_loss, avg_rec, avg_prec, avg_spec = 0., 0., 0., 0.
        avg_f1 = np.zeros((self.annos_.shape[-1],))
        for i in range(N):
            feed_dict_tr = {self.x: np.expand_dims(self.imgs_[i], axis=0),
                            self.y_true: np.expand_dims(self.annos_[i], axis=0),
                            self.rate: 0.,
                            self.is_training: False}

            loss_ = self.sess.run(self.cost_reg, feed_dict=feed_dict_tr)
            f1_ = self.sess.run(self.f1_vec, feed_dict=feed_dict_tr)
            rec_ = self.sess.run(self.recall, feed_dict=feed_dict_tr)
            prec_ = self.sess.run(self.precision, feed_dict=feed_dict_tr)
            spec_ = self.sess.run(self.specificity, feed_dict=feed_dict_tr)

            avg_loss += loss_ / N
            avg_f1 += f1_ / N
            avg_rec += rec_ / N
            avg_prec += prec_ / N
            avg_spec += spec_ / N

            if i % intv == 0:
                if show_pgr is True:
                    utils.show_progress('i '+str(i), loss_, utils.array_to_text(f1_, 3), rec_, prec_, spec_, True)

        # convert f1 vector to text
        avg_f1_txt = utils.array_to_text(avg_f1, 3)

        if show_avg is True:
            utils.show_progress('Avg. results', avg_loss, avg_f1_txt, avg_rec, avg_prec, avg_spec, True)

        return avg_loss, avg_f1


class pruning_system:
    def __init__(self, object):
        self.mdl = object

    def getActivations(self, layer, input, softmax=False):
        """Gets the activations at a given layer for a given input image
        :param layer: layer name
        :param input: input images to the model
        :param softmax: condition to apply softmax"""
        if softmax is True:
            layer = tf.nn.softmax(layer)
        return self.mdl.sess.run(layer, feed_dict={self.mdl.x: input,
                                                   self.mdl.rate: 0.,
                                                   self.mdl.is_training: False})

    def plot_features(self, units, clmns, fig_size, img_name, save, show):
        """Plots activations in a grid
        :param units: feature maps considered
        :param clmns: number of columns
        :param fig_size: figure size
        :param img_name: output image name
        :param save: save results
        :param show: show results"""
        filters = units.shape[3]  # get number of filters used
        # Define plotting grid
        fig = plt.figure(figsize=(fig_size, fig_size))
        n_columns = clmns
        n_rows = math.ceil(filters / n_columns) + 1
        # define output name
        output_name = os.path.join(os.path.abspath(self.mdl.ft_path), img_name)
        plt.axis('off')
        plt.title(img_name)
        for i in range(filters):
            fig.add_subplot(n_rows, n_columns, i + 1)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.get_xaxis().set_visible(False)
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="jet")
        if show is True:
            plt.show()
        if save is True:
            plt.savefig(output_name)
            print(img_name + ' saved.')

    def check_layer(self, n_examples, layer_name, activation='Relu', index=':0', save=False, show=True):
        """Check feature maps of given layer
        :param n_examples: number of input examples whose feature maps are displayed
        :param layer_name: name of the layer considered from the model
        :param activation: activation name
        :param index: index name
        :param save: save results
        :param show: show results
        """
        # Check layer for given blob inputs
        op_name = layer_name
        tensor_name = op_name + '/' + activation + index
        layer = self.mdl.sess.graph.get_tensor_by_name(tensor_name)
        for i in range(n_examples):
            units = self.getActivations(layer=layer,
                                        input=np.expand_dims(self.mdl.imgs_[i], axis=0),
                                        softmax=False)
            # get image number
            nb = self.mdl.img_names[i][0].split('_')[-1].split('.')[0]
            img_name = op_name + '_input-' + nb
            self.plot_features(units, 10, 20, img_name=img_name, save=save, show=show)

    def get_thresholdIndex(self, x, threshold=0.99):
        """Keep indices from sorted vector x whose cumulative sum equals the threshold
        and return index of vector when the threshold is reached to prune remaining indexes"""
        value = 0.
        for i in range(len(x)):
            value += x[i]
            if value >= threshold:
                print('Threshold index: ', i)
                return i

    def norm(self, x):
        """Normalization"""
        return (x - min(x)) / (np.sum(x) - min(x))

    def pruning(self, layer_names, epsilon=0.005, activation='Relu', index=':0'):
        """Prune filters and check the model accuracy recursively
        if change in acc. is > epsilon, then keep the last pruned filter.
        :param layer_names: conv layer names (list)
        :param epsilon: the maximum accuracy change accepted for the model while pruning
        :param activation:  activation function considered
        :param index: tensor index"""
        # get metrics
        print('\nInitial prediction metrics:')
        _, f1 = self.mdl.predict(N=self.mdl.imgs_.shape[0], intv=25, show_avg=True, show_pgr=False)

        count = 0
        current_acc = f1[0]  # used to monitor accuracy change of prediction while pruning
        tokeep_indexes = []  # list filter indices not pruned

        while count < len(layer_names):

            # (1) get the layer considered from the graph
            print('----------------------')
            print('Input: {}'.format(layer_names[count]))
            tensor_name = layer_names[count] + '/' + activation + index
            layer = self.mdl.sess.graph.get_tensor_by_name(tensor_name)

            # (2) calculate eigenvector over many input examples N
            cnt = 0
            vects = []
            N = self.mdl.imgs_.shape[0]
            print('Computing SVDs and pruning indices...')
            for i in range(N):

                if i == cnt * int(N / 4):
                    print('     Processing input...', i)
                    cnt += 1

                # get activation units
                units = self.getActivations(layer,
                                            np.expand_dims(self.mdl.imgs_[i], axis=0),
                                            self.mdl.sess)
                k_sz = units.shape[1]  # kernel size
                f_sz = units.shape[-1]  # filter size

                # reshape into a 2D (non-squared matrix)
                M = units.reshape((k_sz * k_sz, f_sz)).T
                n = M.shape[0]

                # calculate singular value decomposition (SVD)
                # U columns = singular left vectors
                U, Sigma, Vh = np.linalg.svd(M, full_matrices=False, compute_uv=True)

                # take first vector
                vect = U[:, 0]

                # numerate the vector
                ind = np.arange(n)
                vect_ = np.vstack((ind, vect)).T
                # and sort it given its singular values
                vect_ = vect_[np.abs(vect_[:, 1]).argsort()][::-1]

                # remove negative sign for normalization
                vect_n = np.abs(vect_[:, 1])
                vect_n = self.norm(vect_n)

                vects.append(vect_n)

            # take the mean of normalized SV vectors and re-normalize
            vects_mn = np.mean(np.array(vects), axis=0)
            vects_mn = self.norm(vects_mn)

            # (3) get the index of filter beyond which indexes of other filters are pruned
            threshIndex = self.get_thresholdIndex(x=vects_mn, threshold=0.99)
            pruning_indexes = vect_[threshIndex + 1:, 0].astype(np.uint32)

            # (4) Create a mask of 1s and 0s and apply it to the layer's kernel
            # get shape of kernel to prune
            op_name = 'kernel'
            tensor_name = layer_names[count] + '/' + op_name + index
            kernel_ = [v for v in tf.global_variables() if v.name == tensor_name][0]

            # get a copy of kernel to reset it in the network if needed
            kernel_ident = self.mdl.sess.run(kernel_)

            # recursive pruning to reach epsilon
            skip = 0  # increment to skip pruning of last filter considered for pruning is acc. change > epsilon
            while True:
                # skip the first(s) feature map(s)
                pruning_indexes = pruning_indexes[skip:]

                # create the kernel mask (1s) and put 0s where pruned
                mask = np.ones(shape=list(kernel_.shape))  # (h,l,channels,indices)
                mask[:, :, :, pruning_indexes] = 0.

                # apply the mask to prune filters from the layer
                _ = self.mdl.sess.run(tf.assign(kernel_, tf.multiply(kernel_, mask)))

                # (5) get prediction results to check the change in accuracy after pruning
                _, f1 = self.mdl.predict(N=self.mdl.imgs_.shape[0], intv=10, show_avg=False, show_pgr=False)
                change_acc = np.abs(current_acc - f1[0])
                if skip < 1:
                    print('Current | new accuracy: {0:.4f} | {1:.4f} --- '
                          'difference: {2:.4f}'.format(current_acc, f1[0], change_acc))
                current_acc = f1[0]
                # if the accuracy change is more than epsilon
                if change_acc > epsilon:
                    print('     Change in accuracy ({0:.5f}) exceeded. '
                          'Re-iterating the pruning procedure...'.format(change_acc))
                    # reset model's kernel
                    _ = self.mdl.sess.run(tf.assign(kernel_, kernel_ident))
                    skip += 1  # increment skip factor
                else:
                    print('Final accuracy: {0:.5f}'.format(current_acc))
                    print('Pruned filters: ', sorted(pruning_indexes))
                    break
            print('\n')
            # get indexes of filters not pruned
            tokeep_indexes.append(list(set(ind) - set(pruning_indexes)))
            count += 1
