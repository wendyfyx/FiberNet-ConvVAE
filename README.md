# README

### Scripts `/src`

- `data.py` contain 
- `model.py`
- `inference.py`
- `evaluation.py`
- `util.py` contains utility functions.

### Result files `/results`

All data saved from model training is described below

- `/data` contains:
	- training and eval loss as pickle, in the format of `[eval/train]_losses` (tensorboard log is also saved so this is not super helpful);
	- encoded (embeddings), reconstructed and TSNE transformed data, in the format of `[model_name]/E[epoch]_[subject].pkl` where each trained model has its own subfolder.
- `/models` are torch models, in the format of `[model_class]_[Conv_initialization+Linear_initialization]_Z[embedding_dim]_B[batch_size]_LR[learning_rate]_WD[weight_decay]_GC(V/N)[gradient_clip]_E[epochs_trained]_[subject].` 
	- These models are only `state_dict`, which requires the model to be initialized and be used for inference only (not training checkpoint). Follow instructions [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) for saving and loading model. Note the model class and initialize with the appropriate class from `src/model.py`.
	- <u>Initialization</u>: XU for Xavier uniform, XN for Xavier normal, KU for Kaiming uniform, and KN for Kaiming normal; all bias are set to zero regardless of which initialization
	- <u>Gradient</u> clip: GCV for clip gradient value (specifies the `clip_value` parameter in the pytorch function), and GCN for clip gradient norm (default L2 norm, specifies the `max_norm` parameter in the pytorch function)
	- Note that multiple epochs for each model are saved, these can all be loaded for inference.
- `/logs` contains tensorboard log files (for training and eval loss). Each model should have their own folder, so we can compare different loss plots.
	- This is more preferable than the files in `/plots` folder, as we can adjust the range of plots in case there's an extreme value.
- `/plots` contains loss plots in the format of `[model_name]_Loss_[train or eval]_E[total epochs trained]_S[smooth factor]`. This is not super helpful since we have tensorboard logs, but can serve a quick sanity check.
	- Smooth factor of 1 means loss is plotted for each minibatch iteration, value other than 1 usually means it's plotted every epoch.

