# MinEntropy_fn.py
'''
This script is used to calculate the Mapping via the Minimum Entropy Method described by Cnossen2021

The script contains the next functions:
- KL_divergence() 
    the Kullback-Leibler divergence between localization i and j
- Rel_entropy()
    the relative entropy for certain localizations
'''

import tensorflow as tf

#%% functions
@tf.autograph.experimental.do_not_convert
def Rel_entropy(ch1,ch2):
    '''
    Parameters
    ----------
    ch1, ch2 : float32 array 
        The array containing the [x1, x2] locations of all localizations.
    idxlist : list
        List containing per indice of ch1 the neighbours in ch2.

    Returns
    -------
    rel_entropy : float32
        The relative entropy as calculated by Cnossen 2021.

    ''' 
    N = ch1.shape[0]
    expDist = tf.reduce_sum( tf.math.exp(
            -1*KL_divergence(ch1, ch2) / N ) / N
        , axis = 1)
    return -1*tf.reduce_sum(tf.math.log( 
        expDist
        ), axis=0)


@tf.autograph.experimental.do_not_convert
def KL_divergence(ch1, ch2, CRLB = .15):
    '''
    Parameters
    ----------
    ch1, ch2 : 2D float32 array
        The array containing the [x1, x2] locations of the localizations i and j.

    Returns
    -------
    D_KL : float array
        The Kullback Leibler divergence as described by Cnossen 2021.

    '''
    return 0.5*tf.reduce_sum( tf.square(ch1 - ch2) / CRLB**2 , axis=2)