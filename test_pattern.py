from __future__ import print_function
from numpy.testing import (assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal)
import numpy as np
#import nibabel as nib
#import nipy as nip
from nipy.core import image
from nipy.core.reference import coordinate_map as cmap
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

import fmri_qc as fqc

#import sys
#import os
#import tempfile



# set random seed to make the test reproducible 
np.random.seed(42)

def give_pos(pos, N):
    if pos == 'm':
        x =  round(N/2)
    elif pos == 's':
        x = 0
    elif pos == 'e':
        x = N-1
    elif isinstance(pos, int):
        if pos < N and pos >= 0:
            x = pos
    else:
        print("pos, N", pos, N)
        raise ValueError, pos

    return x

def make_pattern(patt, tpos='m', sl='m', sh=(17,5)):

    d = np.zeros(sh, dtype=np.int16)
    T,S = sh
    t = give_pos(tpos, T)
    s = give_pos(sl, S)

    patt = np.asarray(patt)
    assert len(patt.shape) == 1
    lpatt = patt.shape[0]
    
    try:
        if tpos == 'e':
            d[t-lpatt+1:t+1, s] = patt
        else:
            d[t:t+lpatt, s] = patt
    except ValueError:
        print(" patt, tpos, sh ", patt, tpos, sh)

    return d

def test_detect_pattern():

    patt = [1,1,0]
    dshape = (7,3)
    d = make_pattern(patt, 's', 's', sh=dshape)
    assert_equal(d[0:3,0], patt)
    assert_equal(d.shape, dshape)
    T,S = d.shape

    hits = detect_pattern(d, patt, ppos='s', dpos=0)

    assert_equal(hits.shape, (T,S))
    assert hits[0,0] == 1
    assert hits[1,0] == 0
    assert hits[0,1] == 0


def detect_pattern(d, patt, ppos=None, dpos=0):
    """
    d : numpy array  
        The data timeXslice
    patt: numpy array | list
        the pattern to detect
    ppos: 's'|'e'|integer
        a specific position in time to detect the pattern
    dpos: integer
        where to put '1' or 'True' in the result array 
        when pattern is detected (0: start of pattern)
    """
    T,S = d.shape
    hits = np.zeros(d.shape, dtype=np.bool)
    
    patt = np.asarray(patt)
    print(patt)
    assert len(patt.shape) == 1
    lpatt = patt.shape[0]
    assert dpos < lpatt

    lh = T-(lpatt-1)
    # if length of patt is 3, then length of possible hits is T-2

    h = np.ones((lh,S), dtype=np.bool)
    
    for i,p in enumerate(patt):
        print("i:", i, " p:", p)
        d_is_p = (d[i:i+lh,:]==p)
        print(d_is_p)
        print(h)
        print("and")
        h = np.logical_and(h, d_is_p)
        print(h)

    hits[dpos:dpos+lh,:] = h
    print(hits)
    print(d)

    return hits
    

