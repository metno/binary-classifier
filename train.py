#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
# vi: syntax=python


import dataset
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import datetime
import re
from numpy.random import seed
from tensorflow import set_random_seed
import argparse
import os
import tensorflow.contrib.slim as slim;
import sys

# This script was initially from a cv-tricks.com tutorial
# It has a MIT licence

# Hyper params
BATCH_SIZE        = 512


#DROPOUT_KEEP_PROB = 0.22
DROPOUT_KEEP_PROB = 0.3

DO_DROPOUT_ON_HIDDEN_LAYER = True
DROPOUT_KEEP_PROB_HIDDEN = 0.98

# Slow ?
LEARNING_RATE     = 1e-5

# Train/validation split 30% of the data will automatically be used for validation
VALIDATION_SIZE = 0.20

LAMBDA = 0.1
use_L2_Regularization = False

# L2 regularization. This is a good penalty parameter value to start with ?
USE_BATCH_NORMALIZATION = False



parser = argparse.ArgumentParser(description='Train a cnn for predicting cloud coverage')
parser.add_argument('--imagedir', type=str, help='The training and validation data')
parser.add_argument('--outputdir', type=str, default='modeldata', help='where to write model snapshots')
parser.add_argument('--inputdir', type=str, default=None, help='Start training on exising model')

parser.add_argument('--epoch', type=str, default=None, help='Start training from epoch')


parser.add_argument('--logdir', type=str, default='/tmp/tf', help='Metrics data')
args = parser.parse_args()

logs_path = args.logdir


def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


# For tensorboard
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


keep_prob_hidden = tf.placeholder_with_default(1.0, shape=(), name='keep_prob_hidden')
def create_convolutional_layer(
        is_train,
        input,
        num_input_channels,
        conv_filter_size,
        num_filters):


    # Define the weights that will be trained.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    #variable_summaries(weights)

    ## Create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
    #variable_summaries(biases)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                     filter=weights,
                     strides=[1, 1, 1, 1],
                     padding='SAME')

    layer += biases

    ## We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')

    ## Output of pooling is fed to Relu which is the activation function for us.

    if USE_BATCH_NORMALIZATION:
        # One can set updates_collections=None to force the updates in place,
        # but that can have a speed penalty, especially in distributed settings.
        layer = tf.contrib.layers.batch_norm(layer, scale=True, is_training=is_train, zero_debias_moving_mean=True, decay=0.999, updates_collections=None )

    #layer = tf.nn.relu(layer)
    layer = tf.nn.relu(features=layer)

    if DO_DROPOUT_ON_HIDDEN_LAYER == True :
        layer = tf.nn.dropout(layer, keep_prob_hidden)

    return layer


def create_flatten_layer(layer):
    # We know that the shape of the layer will be [BATCH_SIZE img_size img_size num_channels]
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    weights,
                    biases,
                    use_relu=True):

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices,
    # we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        #layer = tf.nn.relu(layer, name='activation')
        layer = tf.nn.relu(features=layer)

    tf.summary.histogram('activations', layer)

    return layer


def show_progress(iteration, epoch, acc_tr, loss_tr, acc_valid, loss_valid):
    msg = "Iteration {5} Training Epoch {0} - Training Accuracy: {1:>6.1%}, Train loss: {2:>.3f}, Validation Accuracy: {3:>6.1%},  Val Loss: {4:.3f}"
    print("%s %s" % (msg.format(epoch + 1, acc_tr, loss_tr, acc_valid, loss_valid, iteration ), datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))


def train(start, num_iterations):

    for i in range(start, num_iterations):
        x_batch, y_true_batch = data.train.next_batch(BATCH_SIZE)
        x_valid_batch, y_valid_batch = data.valid.next_batch(BATCH_SIZE)
        feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch,
                        keep_prob: DROPOUT_KEEP_PROB,
                        keep_prob_hidden: DROPOUT_KEEP_PROB_HIDDEN,
                        is_train: True
        }
        feed_dict_val = {x: x_valid_batch,
                         y_true: y_valid_batch,
                         is_train: False
        }

        # Train:
        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/BATCH_SIZE) == 0:
            epoch = int(i / int(data.train.num_examples/BATCH_SIZE))

            # Calculate training loss and training accuracy
            loss_tr, acc_tr, summary_tr = session.run([cost, accuracy, merged],
                                                      feed_dict=feed_dict_tr)
            # For tensorboard:
            train_writer.add_summary(summary_tr, i)

            # Calculate validation loss and validation accuracy
            loss_valid, acc_valid, summary_val = session.run([cost, accuracy, merged],
                                                             feed_dict=feed_dict_val)

            # Tensorboard:
            test_writer.add_summary(summary_val, i)

            show_progress(i, epoch, acc_tr, loss_tr, acc_valid, loss_valid)

            saver.save(session, args.outputdir + '/binary-predictor-model', global_step=epoch)



if __name__ == "__main__":

    if args.epoch is not None: # If set we continue training from where we left
        os.system("rm -rf /tmp/tf")
    retval = os.system("mkdir -p " + args.outputdir)
    if retval != 0:
        sys.stderr.write('Could not create outputdir\n')
        sys.exit(63)

    print("BATCH_SIZE: %d" % BATCH_SIZE)
    print("DROPOUT_KEEP_PROB %f" % DROPOUT_KEEP_PROB)
    print("LEARNING_RATE: %f" % LEARNING_RATE)
    # Train/validation split 30% of the data will automatically be used for validation
    print("VALIDATION_SIZE: %f" %  VALIDATION_SIZE)
    print("L2 LAMBDA: %f" %  LAMBDA)

    #Adding Seed so that random initialization is consistent
    seed(1)
    set_random_seed(2)

    #Prepare input data
    classes = [0, 1]
    num_classes = len(classes)


    img_size = 128
    num_channels = 3

    # We shall load all the training and validation images and labels into memory
    # using openCV and use that during training
    #data = dataset.read_train_sets(args.labelsfile, args.imagedir, img_size, classes, validation_size=VALIDATION_SIZE)
    data = dataset.read_train_sets(args.imagedir, img_size, classes, validation_size=VALIDATION_SIZE)
    # Shapes of training set
    print("Training set (images) shape: {shape}".format(shape=data.train.images.shape))
    print("Training set (labels) shape: {shape}".format(shape=data.train.labels.shape))

    # Shapes of test set
    print("Test set (images) shape: {shape}".format(shape=data.valid.images.shape))
    print("Test set (labels) shape: {shape}".format(shape=data.valid.labels.shape))



    print("Complete reading input data. ")
    print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
    print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))
    print("data.train.num_examples: %d" % data.train.num_examples)

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=5,
        inter_op_parallelism_threads=5)
    session = tf.Session(config=session_conf)

    # GOLANG note that we must label the input-tensor! (name='x')
    x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')

    ## labels
    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, axis=1)


    is_train = tf.placeholder(tf.bool, name="is_training")

    print("INPUT: ")
    print(x)
    layer_conv1 = create_convolutional_layer(
        is_train,
        input=x,
        num_input_channels=3,
        #conv_filter_size=128,
        conv_filter_size=8,
        num_filters=3
    )
    print("Conv1")
    print(layer_conv1)

    layer_conv2 = create_convolutional_layer(
        is_train,
        input=layer_conv1,
        num_input_channels=3,
        #conv_filter_size=64,
        conv_filter_size=16,
        num_filters=3
    )

    layer_conv3= create_convolutional_layer(
        is_train,
        input=layer_conv2,
        num_input_channels=3,
        conv_filter_size=32,
        num_filters=3
    )

    layer_conv4= create_convolutional_layer(
        is_train,
        input=layer_conv3,
        num_input_channels=3,
        #conv_filter_size=16,
        conv_filter_size=64,
        num_filters=3
    )

    layer_conv5= create_convolutional_layer(
        is_train,
        input=layer_conv4,
        num_input_channels=3,
        #conv_filter_size=8,
        conv_filter_size=128,
        num_filters=3
    )


    layer_flat = create_flatten_layer(layer_conv5)

    #Let's define trainable weights and biases for the fully connected layer1.
    num_inputs=layer_flat.get_shape()[1:4].num_elements()
    #num_outputs=128
    num_outputs=2048
    fc1_weights = create_weights(shape=[num_inputs, num_outputs])
    fc1_biases = create_biases(num_outputs)
    layer_fc1 = create_fc_layer(input=layer_flat,
                                weights=fc1_weights,
                                biases=fc1_biases,
                                use_relu=True
    )

    # Remember: Dropout should only be introduced during training, not evaluation,
    # otherwise your evaluation results would be stochastic as well.
    # Argument to droupout is the probability of _keeping_ the neuron:
    keep_prob = tf.placeholder_with_default(1.0, shape=(), name='keep_prob')
    dropped = tf.nn.dropout(layer_fc1, keep_prob)

    #num_inputs=128
    num_inputs=2048
    num_outputs=num_classes
    fc2_weights = create_weights(shape=[num_inputs, num_outputs])
    fc2_biases = create_biases(num_outputs)


    layer_fc2 = create_fc_layer(input=dropped,
                                weights=fc2_weights,
                                biases=fc2_biases,
                                use_relu=False
    )
    layer2_elms = layer_fc2.get_shape()[1:4].num_elements()
    print("LAYER2 ELMS: " )
    print(layer2_elms)

    # Softmax is a function that maps [-inf, +inf] to [0, 1] similar as Sigmoid. But Softmax also
    # normalizes the sum of the values(output vector) to be 1.
    y_pred = tf.nn.softmax(layer_fc2,name='y_pred')


    # Logit is a function that maps probabilities [0, 1] to [-inf, +inf].
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2,
                                                               labels=y_true)

    # validation_cost = tf.reduce_mean(cross_entropy)


    # cost = loss
    if use_L2_Regularization: # Loss function using L2 Regularization
        regularizer = tf.nn.l2_loss(fc2_weights)
        cost = tf.reduce_mean(cross_entropy + LAMBDA * regularizer)
    else:
        cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    # SGD + momentum :
    #optimizer = tf.train.MomentumOptimizer(learning_rate=LEARNING_RATE, momentum=0.9).minimize(cost)

    # Note that we must label the infer-operation for use from go!!
    y_pred_cls = tf.argmax(y_pred, axis=1, name="infer")
    # This converge fast and should be good enough for our use. Lets use this.
    # turning it off for testing :
    #correct_prediction = tf.abs(tf.subtract(y_pred_cls, y_true_cls)) <= 1
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)


    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #valid_acc, valid_acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, axis=1), predictions=tf.argmax(y_pred, 1))
    #train_acc, train_acc_op = tf.metrics.accuracy(labels=tf.argmax(y_true, axis=1), predictions=tf.argmax(y_pred, 1))

    # Create a summary to monitor cost tensor
    tf.summary.scalar("Loss", cost)
    # Create a summary to monitor accuracy tensor
    tf.summary.scalar("Accuracy", accuracy)
    #tf.summary.scalar("Valid_Accuracy", valid_acc_op)
    #tf.summary.scalar("Train_Accuracy", train_acc_op)

    #tf.summary.scalar('cross_entropy', cross_entropy)

    # merge all summaries into a single "operation" which we can execute in a session
    merged = tf.summary.merge_all()
    # create log writer object

    train_writer = tf.summary.FileWriter(logs_path + '/train', session.graph)
    test_writer  = tf.summary.FileWriter(logs_path + '/test')

    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())

    model_summary()
    saver = tf.train.Saver(max_to_keep=100000)
    path = args.inputdir
    start = 0

    #if path is not None and tf.train.latest_checkpoint(path) is not None:
    if path is not None and args.epoch is not None:
        print("Loading %s  %s " % (path, path + "/cc-predictor-model-" + args.epoch))
        print("Try restoring model ..")
        saver.restore(session, path + "/cc-predictor-model-" + args.epoch)
        print("Training from epoch %d" % int(args.epoch))
        start = int(args.epoch)  * int(data.train.num_examples/BATCH_SIZE) + 2
        print("StartIter: %d " % start)
    train(start, num_iterations=100000000)
