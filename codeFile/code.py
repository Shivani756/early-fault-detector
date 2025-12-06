# Set RECALCULATE_PEAK_FEATURES to True to recalculate the per peak features
RECALCULATE_PEAK_FEATURES = True
# The PREPROCESSED_DIR is used to load the previous commit's per peak features if 
# RECALCULATE_PEAK_FEATURES is set to False. Note, if the kernel is forked this 
# variable needs to be updated to point to the forked kernel's directory
PREPROCESSED_DIR = '../input/vsb-1st-place-solution'

USE_SIMPLIFIED_VERSION = False

if RECALCULATE_PEAK_FEATURES:
    # If performing preprocessing to calculate the per peak features need a newer version of
    # numba than the latest kaggle docker image
    !conda update -y numba

import copy
import gc
import os
import sys
import warnings

from IPython.core.display import display, HTML
import lightgbm as lgb
from matplotlib import pyplot as plt
import numba
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy.stats
import seaborn as sns 
import sklearn
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support

data_dir = '../input/vsb-power-line-fault-detection'

for p in [scipy, np, numba, sklearn, lgb, pyarrow, pd]:
    print('{}=={}'.format(p.__name__, p.__version__))
    
print()

print(sys.version)

# Tested using versions:
#scipy==1.1.0
#numpy==1.16.2
#numba==0.43.0
#sklearn==0.20.3
#lightgbm==2.2.3
#pyarrow==0.10.0
#pandas==0.23.4

meta_train_df = pd.read_csv(data_dir + '/metadata_train.csv')
meta_test_df = pd.read_csv(data_dir + '/metadata_test.csv')

meta_train_df.head()


@numba.jit(nopython=True)
def flatiron(x, alpha=100., beta=1):
    """
    Flatten signal
    
    Creator: Michael Kazachok
    Source: https://www.kaggle.com/miklgr500/flatiron
    """
    new_x = np.zeros_like(x)
    zero = x[0]
    for i in range(1, len(x)):
        zero = zero*(alpha-beta)/alpha + beta*x[i]/alpha
        new_x[i] = x[i] - zero
    return new_x

@numba.jit(nopython=True)
def drop_missing(intersect,sample):
    """
    Find intersection of sorted numpy arrays
    
    Since intersect1d sort arrays each time, it's effectively inefficient.
    Here you have to sweep intersection and each sample together to build
    the new intersection, which can be done in linear time, maintaining order. 

    Source: https://stackoverflow.com/questions/46572308/intersection-of-sorted-numpy-arrays
    Creator: B. M.
    """
    i=j=k=0
    new_intersect=np.empty_like(intersect)
    while i< intersect.size and j < sample.size:
        if intersect[i]==sample[j]: # the 99% case
            new_intersect[k]=intersect[i]
            k+=1
            i+=1
            j+=1
        elif intersect[i]<sample[j]:
            i+=1
        else : 
            j+=1
    return new_intersect[:k]

@numba.jit(nopython=True)
def _local_maxima_1d_window_single_pass(x, w):
    
    midpoints = np.empty(x.shape[0] // 2, dtype=np.intp)
    left_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    right_edges = np.empty(x.shape[0] // 2, dtype=np.intp)
    m = 0 # Pointer to the end of valid area in allocated arrays

    i = 1 # Pointer to current sample, first one can't be maxima
    i_max = x.shape[0] - 1 # Last sample can't be maxima
    while i < i_max:
        # Test if previous sample is smaller
        if x[i - 1] < x[i]:
            i_ahead = i + 1 # Index to look ahead of current sample

            # Find next sample that is unequal to x[i]
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
                    
            i_right = i_ahead - 1
            
            f = False
            i_window_end = i_right + w
            while i_ahead < i_max and i_ahead < i_window_end:
                if x[i_ahead] > x[i]:
                    f = True
                    break
                i_ahead += 1
                
            # Maxima is found if next unequal sample is smaller than x[i]
            if x[i_ahead] < x[i]:
                left_edges[m] = i
                right_edges[m] = i_right
                midpoints[m] = (left_edges[m] + right_edges[m]) // 2
                m += 1
                
            # Skip samples that can't be maximum
            i = i_ahead - 1
        i += 1

    # Keep only valid part of array memory.
    midpoints = midpoints[:m]
    left_edges = left_edges[:m]
    right_edges = right_edges[:m]
    
    return midpoints, left_edges, right_edges


@numba.jit(nopython=True)
def local_maxima_1d_window(x, w=1):
    """
    Find local maxima in a 1D array.
    This function finds all local maxima in a 1D array and returns the indices
    for their midpoints (rounded down for even plateau sizes).
    It is a modified version of scipy.signal._peak_finding_utils._local_maxima_1d
    to include the use of a window to define how many points on each side to use in
    the test for a point being a local maxima.
    Parameters
    ----------
    x : ndarray
        The array to search for local maxima.
    w : np.int
        How many points on each side to use for the comparison to be True
    Returns
    -------
    midpoints : ndarray
        Indices of midpoints of local maxima in `x`.
    Notes
    -----
    - Compared to `argrelmax` this function is significantly faster and can
      detect maxima that are more than one sample wide. However this comes at
      the cost of being only applicable to 1D arrays.
    """    
        
    fm, fl, fr = _local_maxima_1d_window_single_pass(x, w)
    bm, bl, br = _local_maxima_1d_window_single_pass(x[::-1], w)
    bm = np.abs(bm - x.shape[0] + 1)[::-1]
    bl = np.abs(bl - x.shape[0] + 1)[::-1]
    br = np.abs(br - x.shape[0] + 1)[::-1]

    m = drop_missing(fm, bm)

    return m


@numba.jit(nopython=True)
def plateau_detection(grad, threshold, plateau_length=5):
    """Detect the point when the gradient has reach a plateau"""
    
    count = 0
    loc = 0
    for i in range(grad.shape[0]):
        if grad[i] > threshold:
            count += 1
        
        if count == plateau_length:
            loc = i - plateau_length
            break
            
    return loc



#@numba.jit(nopython=True)
def get_peaks(
    x, 
    window=25,
    visualise=False,
    visualise_color=None,
):
    """
    Find the peaks in a signal trace.
    Parameters
    ----------
    x : ndarray
        The array to search.
    window : np.int
        How many points on each side to use for the local maxima test
    Returns
    -------
    peaks_x : ndarray
        Indices of midpoints of peaks in `x`.
    peaks_y : ndarray
        Absolute heights of peaks in `x`.
    x_hp : ndarray
        An absolute flattened version of `x`.
    """
    
    x_hp = flatiron(x, 100, 1)

    x_dn = np.abs(x_hp)
    
    peaks = local_maxima_1d_window(x_dn, window)
    
    heights = x_dn[peaks]
    
    ii = np.argsort(heights)[::-1]
    
    peaks = peaks[ii]
    heights = heights[ii]
    
    ky = heights
    kx = np.arange(1, heights.shape[0]+1)
    
    conv_length = 9

    grad = np.diff(ky, 1)/np.diff(kx, 1)
    grad = np.convolve(grad, np.ones(conv_length)/conv_length)#, mode='valid')
    grad = grad[conv_length-1:-conv_length+1]
    
    knee_x = plateau_detection(grad, -0.01, plateau_length=1000)
    knee_x -= conv_length//2
    
    if visualise:
        plt.plot(grad, color=visualise_color)
        plt.axvline(knee_x, ls="--", color=visualise_color)
    
    peaks_x = peaks[:knee_x]
    peaks_y = heights[:knee_x]
    
    ii = np.argsort(peaks_x)
    peaks_x = peaks_x[ii]
    peaks_y = peaks_y[ii]
        
    return peaks_x, peaks_y, x_hp


@numba.jit(nopython=True)
def clip(v, l, u):
    """Numba helper function to clip a value"""
    
    if v < l:
        v = l
    elif v > u:
        v = u
        
    return v





peak_features_names = [
    'ratio_next',
    'ratio_prev',
    'small_dist_to_min',
    'sawtooth_rmse',
]

num_peak_features = len(peak_features_names)

@numba.jit(nopython=True)
def create_sawtooth_template(sawtooth_length, pre_length, post_length):
    """Generate sawtooth template"""
    
    l = pre_length+post_length+1
    
    st = np.zeros(l)
    for i in range(sawtooth_length+1):
        
        j = pre_length+i
        if j < l:
            st[j] = 1 - ((2./sawtooth_length) * i)
        
    return st

@numba.jit(nopython=True)
def calculate_peak_features(px, x_hp0, ws=5, wl=25):
    """
    Calculate features for peaks.
    Parameters
    ----------
    px : ndarray
        Indices of peaks.
    x_hp0 : ndarray
        The array to search.
    ws : np.int
        How many points on each side to use for small window features
    wl : np.int
        How many points on each side to use for large window features
    Returns
    -------
    features : ndarray
        Features calculate for each peak in `x_hp0`.
    """
    
    features = np.ones((px.shape[0], num_peak_features), dtype=np.float64) * np.nan
    
    for i in range(px.shape[0]):
        
        feature_number = 0
        
        x = px[i]
        x_next = x+1
        x_prev = x-1
        
        h0 = x_hp0[x]

        ws_s = clip(x-ws, 0, 800000-1)
        ws_e = clip(x+ws, 0, 800000-1)
        wl_s = clip(x-wl, 0, 800000-1)
        wl_e = clip(x+wl, 0, 800000-1)
        
        ws_pre = x - ws_s
        ws_post = ws_e - x
        
        wl_pre = x - wl_s
        wl_post = wl_e - x
        
        if x_next < 800000:
            h0_next = x_hp0[x_next]
            features[i, feature_number] = np.abs(h0_next)/np.abs(h0)
        feature_number += 1
            
        if x_prev >= 0:
            h0_prev = x_hp0[x_prev]
            features[i, feature_number] = np.abs(h0_prev)/np.abs(h0)
        feature_number += 1
            
        x_hp_ws0 = x_hp0[ws_s:ws_e+1]
        x_hp_wl0 = x_hp0[wl_s:wl_e+1]
        x_hp_wl0_norm = (x_hp_wl0/np.abs(h0))
        x_hp_ws0_norm = (x_hp_ws0/np.abs(h0))
        x_hp_abs_wl0 = np.abs(x_hp_wl0)
        wl_max_0 = np.max(x_hp_abs_wl0)
        
        ws_opp_peak_i = np.argmin(x_hp_ws0*np.sign(h0))
        
        features[i, feature_number] = ws_opp_peak_i - ws
        feature_number += 1
        
        x_hp_wl0_norm_sign = x_hp_wl0_norm * np.sign(h0)
        
        sawtooth_length = 3
        st = create_sawtooth_template(sawtooth_length, wl_pre, wl_post)
        assert np.argmax(st) == np.argmax(x_hp_wl0_norm_sign)
        assert st.shape[0] == x_hp_wl0_norm_sign.shape[0]
        features[i, feature_number] = np.mean(np.power(x_hp_wl0_norm_sign - st, 2))
        feature_number += 1

        if i == 0:
                assert feature_number == num_peak_features
        
      return features



def process_signal(
    data,
    window=25,
):
    """
    Process a signal trace to find the peaks and calculate features for each peak.
    Parameters
    ----------
    data : ndarray
        The array to search.
    window : np.int
        How many points on each side to use for the local maxima test
    Returns
    -------
    px0 : ndarray
        Indices for each peak in `data`.
    height0 : ndarray
        Absolute heaight for each peak in `data`.
    f0 : ndarray
        Features calculate for each peak in `data`.
    """
    
    px0, height0, x_hp0 = get_peaks(
        data.astype(np.float),
        window=window, 
    )
            
    f0 = calculate_peak_features(px0, x_hp0)
    
    return px0, height0, f0



def process_measurement_peaks(data, signal_ids):
    """
    Process three signal traces in measurment to find the peaks
    and calculate features for each peak.
    Parameters
    ----------
    data : ndarray
        Signal traces.
    signal_ids : ndarray
        Signal IDs for each of the signal traces in measurment
    Returns
    -------
    res : ndarray
        Data for each peak in the three traces in `data`.
    sigid_res : ndarray
        Signal ID for each row in `res`.
    """
    res = []
    sigid_res = []
    
    assert data.shape[1] % 3 == 0
    N = data.shape[1]//3
    
    for i in range(N):
        
        sigids = signal_ids[i*3:(i+1)*3]
        x = data[:, i*3:(i+1)*3].astype(np.float)
        
        px0, height0, f0 = process_signal(
            x[:, 0]
        )
        
        px1, height1, f1 = process_signal(
            x[:, 1]
        )
        
        px2, height2, f2 = process_signal(
            x[:, 2]
        )
        
        if px0.shape[0] != 0:
            res.append(np.hstack([
                px0[:, np.newaxis], 
                height0[:, np.newaxis],
                f0,
 ]))
            
            sigid_res.append(np.ones(px0.shape[0], dtype=np.int) * sigids[0])
        
        if px1.shape[0] != 0:
            res.append(np.hstack([
                px1[:, np.newaxis], 
                height1[:, np.newaxis],
                f1,
            ]))

            sigid_res.append(np.ones(px1.shape[0], dtype=np.int) * sigids[1])
        
        if px2.shape[0] != 0:
            res.append(np.hstack([
                px2[:, np.newaxis], 
                height2[:, np.newaxis],
                f2,
            ]))

            sigid_res.append(np.ones(px2.shape[0], dtype=np.int) * sigids[2])
            
    return res, sigid_res


def process_measurement(data_df, meta_df, fft_data):
    """
    Process three signal traces in measurment to find the peaks
    and calculate features for each peak.
    Parameters
    ----------
    data_df : pandas.DataFrame
        Signal traces.
    meta_df : pandas.DataFrame
        Meta data for measurement
    fft_data : ndarray
        50Hz fourier coefficient for three traces
    Returns
    -------
    peaks : pandas.DataFrame
        Data for each peak in the three traces in `data`.
    """
    peaks, sigids = process_measurement_peaks(
        data_df.values, # [:, :100*3], 
        meta_df['signal_id'].values, # [:100*3]
    )
    
    peaks = np.concatenate(peaks)

    peaks = pd.DataFrame(
        peaks,
        columns=['px', 'height'] + peak_features_names
    )
    peaks['signal_id'] = np.concatenate(sigids)

    # Calculate the phase resolved location of each peak

    phase_50hz = np.angle(fft_data, deg=False) # fft_data[:, 1]

    phase_50hz = pd.DataFrame(
        phase_50hz,
        columns=['phase_50hz']
    )
    phase_50hz['signal_id'] = meta_df['signal_id'].values

    peaks = pd.merge(peaks, phase_50hz, on='signal_id', how='left')

    dt = (20e-3/(800000))
    f1 = 50
 w1 = 2*np.pi*f1
    peaks['phase_aligned_x'] = (np.degrees(
        (w1*peaks['px'].values*dt) + peaks['phase_50hz'].values
    ) + 90) % 360
    
    # Calculate the phase resolved quarter for each peak
    peaks['Q'] = pd.cut(peaks['phase_aligned_x'], [0, 90, 180, 270, 360], labels=[0, 1, 2, 3])
    
    return peaks


@numba.jit(nopython=True, parallel=True)
def calculate_50hz_fourier_coefficient(data):
    """Calculate the 50Hz Fourier coefficient of a signal.
    Assumes the signal is 800000 data points long and covering 20ms.
    """

    n = 800000
    assert data.shape[0] == n
    
    omegas = np.exp(-2j * np.pi * np.arange(n) / n).reshape(n, 1)
    m_ = omegas ** np.arange(1, 2)
    
    m = m_.flatten()

    res = np.zeros(data.shape[1], dtype=m.dtype)
    for i in numba.prange(data.shape[1]):
        res[i] = m.dot(data[:, i].astype(m.dtype))
            
    return res



%%time

if RECALCULATE_PEAK_FEATURES:

    train_df = pq.read_pandas(
        data_dir + '/train.parquet'
    ).to_pandas()
    
    train_fft = calculate_50hz_fourier_coefficient(train_df.values)
    
    train_peaks = process_measurement(
        train_df, 
        meta_train_df, 
        train_fft
    )

    del train_df, train_fft
    gc.collect()



if not RECALCULATE_PEAK_FEATURES:
    train_peaks = pd.read_pickle(PREPROCESSED_DIR + '/train_peaks.pkl')
    
train_peaks.to_pickle("train_peaks.pkl")



train_peaks = pd.merge(train_peaks, meta_train_df[['signal_id', 'id_measurement', 'target']], on='signal_id', how='left')



%%time

if RECALCULATE_PEAK_FEATURES:
    
    NUM_TEST_CHUNKS = 10
    
    test_chunk_size = int(np.ceil((meta_test_df.shape[0]/3.)/float(NUM_TEST_CHUNKS))*3.)
    
    test_peaks = []

    for j in range(NUM_TEST_CHUNKS):

        j_start = j*test_chunk_size
        j_end = (j+1)*test_chunk_size
        
        signal_ids = meta_test_df['signal_id'].values[j_start:j_end]

        test_df = pq.read_pandas(
            data_dir + '/test.parquet',
            columns=[str(c) for c in signal_ids]
        ).to_pandas()

        test_fft = calculate_50hz_fourier_coefficient(test_df.values)
        
        p = process_measurement(
            test_df, 
            meta_test_df.iloc[j_start:j_end], 
            test_fft
        )

        test_peaks.append(p)
        
        print(j)
        
        del test_df
        gc.collect()
        
    del test_fft
    gc.collect()
        
    test_peaks = pd.concat(test_peaks)
    
# Wall time: 9min 45s



if not RECALCULATE_PEAK_FEATURES:
    test_peaks = pd.read_pickle(PREPROCESSED_DIR + '/test_peaks.pkl')
    
test_peaks.to_pickle("test_peaks.pkl")



test_peaks = pd.merge(test_peaks, meta_test_df[['signal_id', 'id_measurement']], on='signal_id', how='left')



def save_importances(importances_, filename='importances.png', print_results=False, figsize=(8, 6)):
    mean_gain = importances_[['gain', 'feature']].groupby('feature').mean()
    if print_results:
        display(HTML(mean_gain.sort_values('gain').to_html()))
    importances_['mean_gain'] = importances_['feature'].map(mean_gain['gain'])
    plt.figure(figsize=figsize)
    sns.barplot(x='gain', y='feature', data=importances_.sort_values('mean_gain', ascending=False))
    plt.tight_layout()
    plt.savefig(filename)



def process(peaks_df, meta_df, use_improved=True):

    results = pd.DataFrame(index=meta_df['id_measurement'].unique())
    results.index.rename('id_measurement', inplace=True)
    
    ################################################################################
    
    if not USE_SIMPLIFIED_VERSION:
        # Filter peaks using ratio_next and height features
        # Note: may not be all that important
        peaks_df = peaks_df[~(
            (peaks_df['ratio_next'] > 0.33333)
            & (peaks_df['height'] > 50)
        )]
    
    ################################################################################

    # Count peaks in phase resolved quarters 0 and 2
    p = peaks_df[peaks_df['Q'].isin([0, 2])].copy()
    res = p.groupby('id_measurement').agg(
    {
        'px': ['count'],
    })
    res.columns = ["peak_count_Q02"]
    results = pd.merge(results, res, on='id_measurement', how='left')
        
    ################################################################################
    
    # Count total peaks for each measurement id
    res = peaks_df.groupby('id_measurement').agg(
    {
        'px': ['count'],
    })
    res.columns = ["peak_count_total"]
    results = pd.merge(results, res, on='id_measurement', how='left')

    ################################################################################

    # Count peaks in phase resolved quarters 1 and 3
    p = peaks_df[peaks_df['Q'].isin([1, 3])].copy()
    res = p.groupby('id_measurement').agg(
    {
        'px': ['count'],
 })
    res.columns = ['peak_count_Q13']
    results = pd.merge(results, res, on='id_measurement', how='left')
    
    ################################################################################
    
    # Calculate additional features using phase resolved quarters 0 and 2
    
    feature_quarters = [0, 2]
    
    p = peaks_df[peaks_df['Q'].isin(feature_quarters)].copy()
    
    p['abs_small_dist_to_min'] = np.abs(p['small_dist_to_min'])
    
    res = p.groupby('id_measurement').agg(
    {
        
        'height': ['mean', 'std'],
        'ratio_prev': ['mean'],
        'ratio_next': ['mean'],
        'abs_small_dist_to_min': ['mean'],        
        'sawtooth_rmse': ['mean'],
    })
    res.columns = ["_".join(f) + '_Q02' for f in res.columns]     
    results = pd.merge(results, res, on='id_measurem