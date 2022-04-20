# FiberNet ConvVAE

### Running the Models

Two environment file (`environment-cpu.yml` and `environment-cuda.yml` ) are provided. Run `conda env create -f environment-[device].yml` to create the appropriate environment.

Follow the `ConvVAE-train.ipynb` and `ConvVAE-inference.ipynb` in `/notebooks` on how to run the ConvVAE model on streamline data.

---

### Scripts `/src`

- `/data/SubjData.py` contains code for loading in bundles (.trk) for a subject. An example data format compatible with DIPY packages can be found [here](https://github.com/dipy/dipy/blob/master/doc/interfaces/buan_flow.rst). For arguments, see comments for function `load_streamlines()` and `load_bundles` in `/data/data_util.py`.

	Example of loading data:

	```python
	args = {'n_points' : 256, 'n_lines' : None, 'min_lines' : 2, 
	        'tracts_exclude' : ['CST_L_s', 'CST_R_s'],'preprocess' : '3d', 
	        'rng' : np.random.RandomState(2022), 'verbose': False, 
	        'data_folder' : '../data/'}
	
	subj_train = SubjData('SubjID001', **args)
	```

- `/model` contains files for model definition `model.py` and training `train_model.py` 

	- `model.py` contains multiple pairs of Encoder and Decoder structure (different number of layers and kernel sizes), to be used with `ConvVAE`. To initalize the model use

		```
		model = init_model(model_type='3Ls', Z=2, SEED=2021)
		```

	- `train_model.py` contains code for training model. Check `ConvVAE-train.ipynb` for usage.

- `inference.py` is for model inference on new data (check `ConvVAE-inference.ipynb` for usage).

- `evaluation.py` contains functions for plotting embeddings, and interpolation.

- `/utils` contains general utility functions.

---

### Result files `/results`

Run `setup.sh` to create the folders. All data saved from model training/inference is described below.

- `/data` contains:
	- Embeddings (`X_encoded`), reconstructed streamlines (`X_recon`) and TSNE transformed data (`X_encoded_tsne` if `apply_tsne=True` during inference), in the format of `[model_name]/E[epoch]_[subject].pkl` where each trained model has its own subfolder.
- `/models` are pytorch models, in the format of `[model_class]_[Conv_initialization+Linear_initialization]_Z[embedding_dim]_B[batch_size]_LR[learning_rate]_WD[weight_decay]_GC(V/N)[gradient_clip]_E[epochs_trained]_[subject].` 
	- These models are only `state_dict`, which requires the model to be initialized and be used for inference only (not training checkpoint). Follow instructions [here](https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference) for saving and loading model. Note the model class and initialize with the appropriate class from `src/model.py`.
	- <u>Initialization</u>: XU for Xavier uniform, XN for Xavier normal, KU for Kaiming uniform, and KN for Kaiming normal; all bias are set to zero regardless of which initialization
	- <u>Gradient</u> clip: GCV for clip gradient value (specifies the `clip_value` parameter in the pytorch function), and GCN for clip gradient norm (default L2 norm, specifies the `max_norm` parameter in the pytorch function)
	- Note that multiple epochs for each model are saved, these can all be loaded for inference.
	- Use `parse_model_setting(model_name)` to get the hyperparameters as dict.
- `/logs` contains tensorboard log files (for training and eval loss). Each model has their own folder, so we can compare different loss plots.
