import torch
import argparse
import numpy as np

from data.SubjData import SubjData
from evaluation import *
from utils.general_util import *
from data.data_util import *
from model.model import *

def _apply_inference(model, subj_data, 
                    mean, std, 
                    model_name, 
                    epoch,
                    result_data_folder="../results/data/",
                    apply_tsne=True, 
                    perplexity=100,
                    save_result=True):
    result = {}
    
    X_norm = subj_data.X_norm
    X_encoded, X_recon = get_reconstruction_vae(model, X_norm, mean=mean, std=std)
    result["X_encoded"] = X_encoded
    result["X_recon"] = X_recon
        
    if apply_tsne:
        print("Applying TSNE...")
        X_encoded_tsne = tsne_transform(X_encoded, perplexity=perplexity)
        result[f"X_encoded_tsne"] = X_encoded_tsne
    
    if save_result:
        if not result_data_folder.endswith("/"):
            result_data_folder = result_data_folder + "/"
        result_fpath = f"{result_data_folder}{model_name}/E{epoch}_{subj_data.name}"
        print(f"Saving result to {result_fpath}")
        save_pickle(result, result_fpath)
    
    return result

def load_model(name, model_folder, epoch, device):
    model_info = parse_model_setting(name)
    model = init_model(model_type=model_info['model_type'], 
                        Z=model_info['Z'])
    model.load_state_dict(torch.load(f"{model_folder}{name}/model_E{epoch}", 
                                 map_location=torch.device(device)))
    return model

def apply_inference(args):
    
    # Load model
    model = load_model(args.model_name, args.model_folder, args.epoch, args.device)

    # Define required params
    mean = torch.tensor([  1.1069, -23.8073,  13.3312])
    std = torch.tensor([25.0185, 28.6887, 23.8379])

    data_args = {'n_points' : 256, 'n_lines' : None, 'min_lines' : 2, 
            'tracts_exclude' : ['CST_L_s', 'CST_R_s'], 'preprocess' : '3d', 
            'rng' : np.random.RandomState(args.seed), 'verbose': False, 
            'data_folder' : args.data_folder}

    # load subject data
    subj_data = SubjData(args.subj, **data_args)

    # apply mean and std to new data
    subj_data.X_norm = torch.from_numpy(subj_data.X).sub(mean).div(std)

    # apply inference
    _apply_inference(model, subj_data, 
                    mean, std, 
                    model_name=args.model_name, 
                    epoch=args.epoch,
                    result_data_folder=args.result_data_folder,
                    apply_tsne=args.perplexity,
                    perplexity=args.perplexity,
                    save_result=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--subj', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--perplexity', type=int, required=False)
    parser.add_argument('--data_folder', type=str, required=True)
    parser.add_argument('--model_folder', type=str, required=True)
    parser.add_argument('--result_data_folder', type=str, required=True)

    args = parser.parse_args()
    apply_inference(args)

if __name__ == '__main__':
    main()

