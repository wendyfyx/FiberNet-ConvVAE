import glob
import numpy as np
from data.data_util import *

class SubjData:
    
    @classmethod
    def __init__(self, name, data_folder, **kwargs):
        
        self.name = name
        
        # Load streamlines
        X, bundle_idx = load_bundles(data_folder+name, self.parse_tract_name, **kwargs)
        self.X = X
        self.bundle_idx = bundle_idx # {bundle : [start, bundle_count]}

        # Get bundle labels
        y, bundle_num = make_y(self.bundle_idx) 
        self.y = y
        self.bundle_num = bundle_num # {index : bundle}
        
        print(f"Loaded {self.name} with {len(self.bundle_idx)} tracts and {len(self.X)} lines.")


    @classmethod
    def get_bundle_idx(self, bundle):
        '''
            Get indices of bundle in X
            Example usage: subj.X[subj.get_bundle_idx('V')]
        '''
        return self._get_bundle_idx(bundle, self.bundle_idx)
        
    
    @staticmethod
    def _get_bundle_idx(bundle, bundle_idx):

        if bundle not in bundle_idx:
            print(f"Bundle {bundle} does not exist in this subject.")
            return
        indices = bundle_idx[bundle]
        return np.arange(indices[0], indices[0]+indices[1])

    @staticmethod
    def parse_tract_name(fname):
        return "_".join(fname.split('_')[1:-2])