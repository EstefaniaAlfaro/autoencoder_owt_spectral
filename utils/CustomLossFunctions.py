import tensorflow as tf
import numpy as np


def focal_loss(y_true, y_pred):
    y_true = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    y_pred = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    custom_loss = tf.square((y_true - y_pred) / 10)
    return custom_loss


def spectral_angle_distance(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=-1)
    y_pred = tf.math.l2_normalize(y_pred, axis=-1)
    product_y_true_y_pred = (y_true * y_pred)
    spectral_angle = tf.math.acos(product_y_true_y_pred)
    return spectral_angle


def mean_square_error(y_true, y_pred):
    loss_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    return loss_mse


def total_loss_function(y_true, y_pred):
    total_loss = mean_square_error(y_true, y_pred)
    return total_loss


def spectral_angle_distance_metric(y_true, y_pred):
    product_y_true_y_pred = np.dot(y_true, y_pred)
    y_true_norm = np.linalg.norm(y_true)
    y_pred_norm = np.linalg.norm(y_pred)
    spectral_distance = np.arccos((product_y_true_y_pred / (y_true_norm * y_pred_norm)))
    return spectral_distance


def cross_entropy_loss(y_true, y_pred):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return cross_entropy


def loss(model, original):
    reconstruction_data = spectral_angle_distance(model(original), original)
    return reconstruction_data
