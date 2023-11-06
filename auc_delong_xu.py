#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 14:29:57 2018

@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison
"""
from __future__ import absolute_import, division

import numpy as np
from scipy import stats
import scipy.stats


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """
    Computes midranks.

    Parameters
    ----------
        x : np.array
            x - a 1D numpy array
    Returns
    -------
       T2 : np.array
           Array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1

    return T2


def compute_midrank_weight(x, sample_weight):
    """
    Computes midranks.

    Parameters
    ----------
        x : np.array
        sample_weigh : int
            x - a 1D numpy array
    Returns
    -------
        T2 : np.array
            array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(
            predictions_sorted_transposed,
            label_1_count)
    else:
        return fastDeLong_weights(
            predictions_sorted_transposed,
            label_1_count,
            sample_weight)


def fastDeLong_weights(pred_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Reference
    ----------
    @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }


    Parameters
    ----------
       predictions_sorted_transposed : np.array
           a 2D numpy.array[n_classifiers, n_examples] sorted such as the
           examples with label "1" are first
    Returns
    -------
        aucs : float
        delongcov : float
            (AUC value, DeLong covariance)

    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = pred_sorted_transposed.shape[1] - m
    positive_examples = pred_sorted_transposed[:, :m]
    negative_examples = pred_sorted_transposed[:, m:]
    k = pred_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(
            positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(
            negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(
            pred_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(
        sample_weight[:m, np.newaxis],
        sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (
        sample_weight[:m]*(tz[:, :m] - tx)
    ).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.

    Reference:
        @article{sun2014fast,
             title={
                 Fast Implementation of DeLong's Algorithm for
                 Comparing the Areas Under Correlated Receiver Oerating
                 Characteristic Curves},
             author={Xu Sun and Weichao Xu},
             journal={IEEE Signal Processing Letters},
             volume={21},
             number={11},
             pages={1389--1393},
             year={2014},
             publisher={IEEE}
         }

    Parameters
    ----------
        predictions_sorted_transposed : ?
        label_1_count : ?

            predictions_sorted_transposed: a 2D
            ``numpy.array[n_classifiers, n_examples]``
            sorted such as the examples with label "1" are first
    Returns
    -------
       (AUC value, DeLong covariance)


    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """
    Computes log(10) of p-values.

    Parameters
    ----------
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances

    Returns
    -------
       log10(pvalue)
    """
    l_aux = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l_aux, sigma), l_aux.T))
    log_p_val = np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)
    return log_p_val, z


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Parameters
    ----------
        ground_truth: np.array
            of 0 and 1
        predictions: np.array
            of floats of the probability of being class 1
    """
    ground_truth_stats = compute_ground_truth_statistics(
        ground_truth,
        sample_weight)
    order, label_1_count, ordered_sample_weight = ground_truth_stats

    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed,
        label_1_count,
        ordered_sample_weight)

    assert_msg = "There is a bug in the code, please forward this to the devs"
    assert len(aucs) == 1, assert_msg
    return aucs[0], delongcov


def delong_roc_test(ground_truth, pred_one, pred_two, sample_weight=None):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different

    Parameters
    ----------
       ground_truth: np.array
           of 0 and 1
       predictions_one: np.array
           predictions of the first model,
           np.array of floats of the probability of being class 1
       predictions_two: np.array
           predictions of the second model, np.array of floats of the
           probability of being class 1
    """
    order, label_1_count, _ = compute_ground_truth_statistics(
        ground_truth,
        sample_weight)

    predictions_sorted_transposed = np.vstack(
        (pred_one, pred_two))[:, order]

    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed,
        label_1_count,
        sample_weight)

    log_p_val = calc_pvalue(aucs, delongcov)
    return log_p_val


def auc_ci_Delong(y_true, y_scores, alpha=.95):
    """AUC confidence interval via DeLong.

    Computes de ROC-AUC with its confidence interval via delong_roc_variance

    References
    -----------
        See this `Stack Overflow Question
        <https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals/53180614#53180614/>`_
        for further details

    Examples
    --------

    ::

        y_scores = np.array(
            [0.21, 0.32, 0.63, 0.35, 0.92, 0.79, 0.82, 0.99, 0.04])
        y_true = np.array([0, 1, 0, 0, 1, 1, 0, 1, 0])

        auc, auc_var, auc_ci = auc_ci_Delong(y_true, y_scores, alpha=.95)

        np.sqrt(auc_var) * 2
        max(auc_ci) - min(auc_ci)

        print('AUC: %s' % auc, 'AUC variance: %s' % auc_var)
        print('AUC Conf. Interval: (%s, %s)' % tuple(auc_ci))

        Out:
            AUC: 0.8 AUC variance: 0.028749999999999998
            AUC Conf. Interval: (0.4676719375452081, 1.0)


    Parameters
    ----------
    y_true : list
        Ground-truth of the binary labels (allows labels between 0 and 1).
    y_scores : list
        Predicted scores.
    alpha : float
        Default 0.95

    Returns
    -------
        auc : float
            AUC
        auc_var : float
            AUC Variance
        auc_ci : tuple
            AUC Confidence Interval given alpha

    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Get AUC and AUC variance
    auc, auc_var = delong_roc_variance(
        y_true,
        y_scores)
    #Added "+1.0e-08" to avoid the following error: RuntimeWarning: invalid value encountered in multiply
    auc_std = np.sqrt(auc_var+1.0e-8)

    # Confidence Interval
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    lower_upper_ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    lower_upper_ci[lower_upper_ci > 1] = 1

    return auc, auc_var, lower_upper_ci
