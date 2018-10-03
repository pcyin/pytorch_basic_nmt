import numpy as np
import nltk
import math
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction


_smooth_func = SmoothingFunction().method3


def get_reward(ref_sent, hyp_sent, smooth_func=_smooth_func):
    r = sentence_bleu([ref_sent], hyp_sent, smoothing_function=smooth_func)

    return r


def mcmc_accept(p_y_tm1, q_y_tm1, p_y_t, q_y_t, tau=1.0):
    acc_ratio = math.exp(q_y_t / tau - q_y_tm1 / tau + p_y_tm1 - p_y_t)
    u = np.random.uniform(0, 1)

    return u <= acc_ratio

