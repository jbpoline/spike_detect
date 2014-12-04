from __future__ import print_function
from numpy.testing import (assert_array_almost_equal, assert_almost_equal, assert_array_equal, assert_equal)
import numpy as np
#import nibabel as nib
#import nipy as nip
from nipy.core import image
from nipy.core.reference import coordinate_map as cmap
from nipy.algorithms.diagnostics.timediff import time_slice_diffs

import fmri_qc as fqc


#----------------------- utility functions ------------------
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
    lpatt = len(patt)
    dshape = (7,3)

    d = make_pattern(patt, 's', 's', sh=dshape)

    assert_equal(d[0:3,0], patt)
    assert_equal(d.shape, dshape)
    T,S = d.shape

    d = make_pattern(patt, 's', 's', sh=dshape)
    hits = fqc.detect_pattern(d, patt, ppos='s', dpos=0)

    assert_equal(hits.shape, (T,S))
    assert hits[0,0] == 1
    assert hits[1,0] == 0
    assert hits[2,0] == 0
    assert hits[0,1] == 0

    sl = 1
    d = make_pattern(patt, 's', sl, sh=dshape)
    hits = fqc.detect_pattern(d, patt, ppos='s', dpos=0)

    assert_equal(hits.shape, (T,S))
    assert hits[0,sl] == 1
    assert hits[1,sl] == 0
    assert hits[2,sl] == 0
    assert hits[0,sl-1] == 0

    hits = fqc.detect_pattern(d, patt, ppos='e', dpos=0)

    assert hits[T-lpatt, 0] ==  0, print(hits)
    assert hits[T-lpatt+1,0] == 0
    assert hits[T-lpatt+2,0] == 0

    d = make_pattern(patt, 'e', 's', sh=dshape)
    hits = fqc.detect_pattern(d, patt, ppos='e', dpos=0)

    assert hits[T-lpatt, 0] ==  1, print(hits)
    assert hits[T-lpatt+1,0] == 0
    assert hits[T-lpatt+2,0] == 0

    hits = fqc.detect_pattern(d, patt, ppos='e', dpos=1)

    assert hits[T-lpatt, 0] ==  0, print("\n",hits)
    assert hits[T-lpatt+1,0] == 1, print("\n",hits)
    assert hits[T-lpatt+2,0] == 0, print("\n",hits)

    d = make_pattern(patt, 's', 's', sh=dshape)
    hits = fqc.detect_pattern(d, patt, ppos='s', dpos=1)

    assert_equal(hits.shape, (T,S))
    assert hits[0,0] == 0
    assert hits[1,0] == 1
    assert hits[2,0] == 0

#----------------------- utility functions ------------------

def test_add_histeresis():
    """
    """
    a = np.asarray(  [1.2, .2, 1.3, 0.4, .9, 2, 10, 8, 1.3, 1.1])
        # rank array([  4,  0,   5,   1,  2, 7, 9,  8,   6, 3])
            
    spik = a > 9 # get the max
    hspik = fqc.add_histeresis(a, spik, hthres=2.)

    assert hspik[7] == 1, print("spik:", spik, "\nhspik: ", hspik)
    assert hspik[6] == 0, print("spik:", spik, "\nhspik: ", hspik)
    assert hspik[8] == 0, print("spik:", spik, "\nhspik: ", hspik)

    a = np.asarray(  [1.2, 12, 13, 0.4, .9, 2, 10, 8, 1.3, 1.1])
    spik = a > 9 # get the max
    hspik = fqc.add_histeresis(a, spik, hthres=2.)

    assert hspik[7] == 1, print("spik:", spik, "\nhspik: ", hspik)
    assert hspik[6] == 0, print("spik:", spik, "\nhspik: ", hspik)
    assert hspik[8] == 0, print("spik:", spik, "\nhspik: ", hspik)

    a = np.asarray(  [1.2, .2, 13, 0.4, .9, 2, 10, 8, 1.3, 1.1])
    spik = a > 10 # get the max
    hspik = fqc.add_histeresis(a, spik, hthres=2.)
    assert not np.any(hspik), print(hspik)

def test_spikes_from_slice_diff():
    
    a = np.asarray(  [[1.2, 0.2, 1.3, 0.4, .9, 2, 10,   8, 1.3, 1.1],
                      [1.2,  12, 13,  0.4, .9, 2,  1, 0.8, 1.3, 1.1],
                      [1.2, 1.2, 1.3, 0.4, .9, 2,  1, 0.8, 1.3, 1.1]])
    spik = fqc.spikes_from_slice_diff(a.T, verbose=1)
    print("\n spik \n", spik)
    good =            [[0,0,0,0,0,0,1,1,0,0], 
                       [0,1,1,0,0,0,0,0,0,0], 
                       [0,0,0,0,0,0,0,0,0,0]]
    
    assert_array_equal(spik, np.asarray(good).T)
    



# set random seed to make the test reproducible 
#np.random.seed(142)

def make_data(sh=(4,4,5,17)):
    d = np.random.normal(size=sh)
    #d = d/d.max() # d values between 0 and 1.

    return d


def make_bad_slices(data, sli, tim, offset=4., scale=1.):
    """
    make some bad slices : replace
    eg: make_bad_slices(data, ([0:15], [2, 4, 9]), (2, 10)) 
    does bad slices 0:15 at time 2, and 2,4,9 at time 10
    """

    dx,dy,dz,dt = data.shape

    # check : one set of slice per times
    assert len(sli) == len(tim)
    
    # check the slices values are less than number of slices
    for idx in range(len(sli)):
        assert np.all(np.asarray(sli[idx]) < dz), \
                        print("idx", idx, "sli", sli, "dz", dz)

    # check the time values are less than number of time points 
    assert np.all(np.asarray(tim) < dt)

    for idx,ti in enumerate(tim):
        data[:,:,sli[idx], ti] *= scale
        data[:,:,sli[idx], ti] += offset

    return data


def dummy_bold_img(sh=(4,4,5,17), sli=(range(5), [0, 2, 4]), tim=(2,10)):
    
    d = make_data(sh)
    d = make_bad_slices(d, sli, tim)
    assert len(d.shape) == 4

    aff = cmap.AffineTransform('ijkl','xyzt', np.eye(5))
    img = image.Image(d, aff)

    return img

def one_detection(sh,sli,tim):
    arr = make_data(sh)
    arr = make_bad_slices(arr, sli, tim)
    qc = time_slice_diffs(arr)
    smd2 = qc['slice_mean_diff2']
    spikes = fqc.spikes_from_slice_diff(smd2, Zalph=5., histeresis=True, 
                                            hthres=2., verbose=1)
    print("\n spikes ine one detec\n", spikes)
    final = fqc.final_detection(spikes, verbose=1)
    print("\n final ine one detec\n", final)
    times_to_correct = np.where(final.sum(axis=1) > 0)[0]
    print("times_to_correct: ",times_to_correct)
    slices_to_correct = {}
    for ti in times_to_correct:
        slices_to_correct[ti] = np.where(final[ti,:] > 0)[0]

    return times_to_correct, slices_to_correct

def test_spike_detector():
  
    all_sh  = ((4,4,5,17), (4,4,5,17), (4,4,5,17)) 
    all_sli = ((range(5), [0, 2, 4]), ([3], [0, 2, 4]), ([0,4], [0,4]))
    all_tim = ((2,10), (0,10), (0,16))

    #sh  = (4,4,5,17)
    #sli = ([3],)
    #tim = (0,)

    for sh, sli, tim in zip(all_sh, all_sli, all_tim):
        print("\nsh,sli,tim: \n", sh,sli,tim)
        times_2_correct, slices_2_correct =  one_detection(sh,sli,tim)
        print("times_to_correct: ",times_2_correct, "slices_to_correct: ",
                slices_2_correct)
        assert_array_equal(np.asarray(tim), times_2_correct)
        for idx,ti in enumerate(times_2_correct):
            assert_array_equal(np.asarray(slices_2_correct[ti]), 
                    np.asarray(sli[idx]))



