import glob
import numpy as np
import dipy
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import set_number_of_points, select_random_set_of_streamlines
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


def load_streamlines(fpath, n_points=256, n_lines=None,
                     preprocess=None, rng=None,
                     verbose=True, **kwargs):

    '''Load streamlines from one .trk file (one tract).'''

    lines = load_tractogram(fpath, reference="same", bbox_valid_check=False).streamlines
    if n_points is not None:
        lines = set_number_of_points(lines, n_points)
    if n_lines is not None:
        lines = select_random_set_of_streamlines(lines, n_lines, rng=rng)

    if verbose:
        fname = fpath.split('/')[-1]
        print(f"Loaded {fname} with {len(lines)} lines each with {n_points} points")

    if preprocess == "2d":
        lines = lines.get_data()
        if verbose:
            print(f"Preprocessed lines into {preprocess} with shape {lines.shape}")
    if preprocess == "3d":
        if n_points == None:
            print("Cannot process into 3D if n_points=None, returning ArraySequence")
            return lines
        if n_lines is not None:
            n_lines = min(n_lines, len(lines))
        else:
            n_lines = len(lines)
        lines = lines.get_data().reshape((n_lines, n_points, 3))
        if verbose:
            print(f"Preprocessed lines into {preprocess} with shape {lines.shape}")

    return lines


def load_bundles(folder_path, parse_tract_func, min_lines=2, 
                tracts_exclude=None, sub_folder_path="rec_bundles/",
                **kwargs):
    '''Load bundles in folder, sorted alphabetically by tract name.'''

    lines = []
    bundle_idx = {}

    if not folder_path.endswith("/"):
        folder_path = folder_path + "/"
    if sub_folder_path:
        if not sub_folder_path.endswith("/"):
            sub_folder_path = sub_folder_path + "/"

    lines_count = 0
    for fpath in sorted(list(glob.glob(folder_path + sub_folder_path + "*.trk"))):  
        fname = fpath.split('/')[-1]
        tract = parse_tract_func(fname)
        
        if tracts_exclude:
            if tract in tracts_exclude:
                continue

        bundle = load_streamlines(fpath, **kwargs)
        if len(bundle) < min_lines:
            continue
            
        lines.append(bundle)
        bundle_idx[tract]=[lines_count, len(bundle)]
        lines_count += len(bundle)

    if len(lines)==0:
        print(f"No bundle was loaded from {folder_path}")
        return
    lines = np.concatenate(lines)

    return lines, bundle_idx


def make_y(bundle_idx):
    '''
        Make labels from bundle information. Return 1D array of index, and 
        a dictionary of the corresponding bundle names.
    '''

    y = []
    bundle_num = {}
    
    for idx, bundle in enumerate(sorted(bundle_idx)):
        y.append([idx] * bundle_idx[bundle][1])
        bundle_num[idx] = bundle
    y = np.concatenate(y)
    return y, bundle_num


# def load_subject(subj, n_points=256, n_lines=None, 
#                  preprocess='3d', rng=None,
#                  min_lines = 2,
#                  data_folder="../sample-data/",
#                  verbose=True):

#     '''Load data for one subject, including streamline data and bundle information.'''

#     data_subj = {}
#     X, bundle_count, bundle_idx = load_bundles(data_folder+subj, 
#                                                n_points=n_points, n_lines=n_lines, 
#                                                min_lines = min_lines, preprocess=preprocess, 
#                                                rng=rng, verbose=verbose)
#     y, bundle_num = make_y(bundle_count, verbose=verbose)
#     data_subj['name'] = subj
#     data_subj['X'] = X
#     data_subj['bundle_count'] = bundle_count
#     data_subj['bundle_idx'] = bundle_idx
#     data_subj['y'] = y
#     data_subj['bundle_num']=bundle_num
#     print(f"Loaded {subj} with {len(bundle_count)} tracts and {len(X)} lines.")
#     return data_subj


def split_data(X, y=None, n_splits=10, test_size=0.2, random_state=1):
    '''
    [DATA-UTIL]
    PARAMETERS:
      n_splits: Number of re-shuffling & splitting iterations.
      test_size: Percentage of test data
    RETURN:
      train: numpy array containing index of the train samples in df_x
      test: numpy array containing index of the test samples in df_x
    NOTE:
      To get train-val-test split, call split_data() once on orig data, once on train data
    '''
    X = [1 for i in range(len(X))]
    if y is not None:
        sss = StratifiedShuffleSplit(
            n_splits=n_splits, test_size=test_size, random_state=random_state)
        train, test = next(sss.split(X, y))
    else:
        sss = ShuffleSplit(n_splits=n_splits,
                           test_size=test_size, random_state=random_state)
        train, test = next(sss.split(X))
    print(f"Split into {len(train)} train and {len(test)} test samples")
    return train, test


def get_all_subj(data_folder):
    '''[DATA-UTIL] Get a list of all subjects in a folder'''
    return [subj.split('/')[-1] for subj in \
                sorted(list(glob.glob(f"{data_folder}/*")))]