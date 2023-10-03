import random as python_random

import os
import csv
import random as python_random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample, shuffle
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors

# from _composite import ModifiedLatentCF
from _guided import ModifiedLatentCF
from _vanilla import LatentCF

def reset_seeds(seed_value=39):
    # ref: https://keras.io/getting_started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    # necessary for starting Numpy generated random numbers in a well-defined initial state.
    np.random.seed(seed_value)
    # necessary for starting core Python generated random numbers in a well-defined state.
    python_random.seed(seed_value)
    # set_seed() will make random number generation
    tf.random.set_seed(seed_value)

def upsample_minority_multivariate(X, y, random_state=39):
    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_count = max(class_counts)

    X_resampled_list = []
    y_resampled_list = []

    for cls in unique_classes:
        X_cls = X[y == cls]
        y_cls = y[y == cls]

        # Resample the current class data to match the max count
        X_cls_resampled, y_cls_resampled = resample(X_cls, y_cls,
                                                    replace=True, # sample with replacement (upsample)
                                                    n_samples=max_count, # match the number in majority class
                                                    random_state=random_state) # reproducible results

        X_resampled_list.append(X_cls_resampled)
        y_resampled_list.append(y_cls_resampled)

    # Vertically stack the resampled data for each class
    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.hstack(y_resampled_list)
    X_concat, y_concat = shuffle(X_resampled, y_resampled, random_state=random_state)

    return X_resampled, y_resampled


def normalize_multivariate(data, n_timesteps, n_features, scaler=None):

    # Then reshape data to have timesteps as rows for normalization
    data_reshaped = data.reshape(-1, n_features)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_reshaped)

    normalized = scaler.transform(data_reshaped)

    # Return data reshaped
    data = normalized.reshape(-1, n_timesteps, n_features)
    return data, scaler

def conditional_pad_multivariate(X):
    num_timesteps = X.shape[1]

    if num_timesteps % 4 != 0:
        next_num = (int(num_timesteps / 4) + 1) * 4
        padding_size = next_num - num_timesteps
        X_padded = np.pad(
            X, pad_width=((0, 0), (0, padding_size), (0, 0))
        )

        return X_padded, padding_size

    return X, 0


def validity_score(pred_labels, cf_labels):
    desired_labels = 1 - pred_labels  # for binary classification
    return accuracy_score(y_true=desired_labels, y_pred=cf_labels)

def euclidean_distance(X, cf_samples, average=True):
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    return np.mean(paired_distances) if average else paired_distances

def evaluate(X_pred_neg, best_cf_samples, z_pred, n_timesteps, maximum_distance=1):
    proxi = euclidean_distance(X_pred_neg, best_cf_samples)
    valid = validity_score(z_pred)
    # compact = compactness_score(X_pred_neg, best_cf_samples, n_timesteps=n_timesteps)
    cost_mean, cost_std = cost_score(z_pred)
    
    # neigh_count = neighbour_counts_within_radius(best_cf_samples.reshape(-1, n_timesteps), tree, radius=0.3*maximum_distance)
    # dist_knn = distance_knn(best_cf_samples.reshape(-1, n_timesteps), tree) 

    # return proxi, valid, compact, cost_mean, cost_std, neigh_count, dist_knn     # exclude unused metrics in the final evaluation
    return proxi, valid, cost_mean, cost_std
def compactness_score(X, cf_samples):
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.01)

    # return a positive compactness, instead of 1 - np.mean(..)
    return np.mean(c, axis=(1, 0))
  
def find_best_lr(
    classifier,
    X_samples,
    pred_labels,
    autoencoder=None,
    encoder=None,
    decoder=None,
    lr_list=[0.001, 0.0001],
    pred_margin_weight=1.0,
    step_weights=None,
    random_state=None,
    padding_size=0,
    target_prob=0.5,
):
    # Find the best alpha for vanilla LatentCF
    best_cf_model, best_cf_samples, best_cf_embeddings = None, None, None
    best_losses, best_valid_frac, best_lr = 0, -1, 0

    for lr in lr_list:
        print(f"======================== CF search started, with lr={lr}.")
        # Fit the LatentCF model
        # TODO: fix the class name here: ModifiedLatentCF or GuidedLatentCF? from _guided or _composite?
        if encoder and decoder:
            cf_model = ModifiedLatentCF(
                probability=target_prob,
                only_encoder=encoder,
                only_decoder=decoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )
        else:
            cf_model = ModifiedLatentCF(
                probability=target_prob,
                autoencoder=autoencoder,
                optimizer=tf.optimizers.Adam(learning_rate=lr),
                pred_margin_weight=pred_margin_weight,
                step_weights=step_weights,
                random_state=random_state,
            )

        cf_model.fit(classifier)

        if encoder and decoder:
            cf_embeddings, losses, _ = cf_model.transform(X_samples, pred_labels)
            cf_samples = decoder.predict(cf_embeddings)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_embeddings)
            cf_pred_labels = np.argmax(z_pred, axis=1)
        else:
            cf_samples, losses, _ = cf_model.transform(X_samples, pred_labels)
            # predicted probabilities of CFs
            z_pred = classifier.predict(cf_samples)
            cf_pred_labels = np.argmax(z_pred, axis=1)

        valid_frac = validity_score(pred_labels, cf_pred_labels)
        proxi_score = euclidean_distance(
            remove_paddings(X_samples, padding_size),
            remove_paddings(cf_samples, padding_size),
        )

        # uncomment for debugging
        print(f"lr={lr} finished. Validity: {valid_frac}, proximity: {proxi_score}.")

        # TODO: fix (padding) dimensions of `lof_estimator` and `nn_estimator` during training, for debugging
        # proxi_score, valid_frac, lof_score, rp_score, cost_mean, cost_std = evaluate(
        #     X_pred_neg=X_samples,
        #     cf_samples=cf_samples,
        #     z_pred=z_pred,
        #     n_timesteps=_,
        #     lof_estimator=lof_estimator,
        #     nn_estimator=nn_estimator,
        # )

        # if valid_frac >= best_valid_frac and proxi_score <= best_proxi_score:
        if valid_frac >= best_valid_frac:
            best_cf_model, best_cf_samples = cf_model, cf_samples
            best_losses, best_lr, best_valid_frac = losses, lr, valid_frac
            if encoder and decoder:
                best_cf_embeddings = cf_embeddings

    return best_lr, best_cf_model, best_cf_samples, best_cf_embeddings
