### 0. Download dataset
Download the crohd dataset here:
https://drive.google.com/file/d/11ihxAEyNze6laCGLqQ3LAZQkIQ5FJf7k/view?usp=drive_link


### 1. Pseudo-label generation
Start tracking clips loaded from the crohd dataset for points in a window centered around interested query points (i.e. points you want to track). Trajectories that meet the Cycle Consistency check will be saved and kept, as described in the paper.

Run:
```
python3 get_pseduolabel_crohd.py --dataset_path [CROHD_DATASET_DIR] --dset_split [DSET_SPLIT]
```

Visualizations are also logged during this process for you to perform sanity check on the generated pseudo-labels.

DSET_SPLIT: Since this process could take a while, DSET_SPLIT is used to split the entire dataset into six roughly equal pieces to allow the generation to run parallely on each split simultanously.

To get data for all splits, simply run:
```
bash get_pseduolabel_crohd.sh
```
Refer to get_pseduolabel_crohd.sh for how to do this parallely with non-blocking command if you have multiple GPUs.

### 1.5 Aggregate and process generated data
Aggregate the pseudo-labels generated from all videos in the above step. 

```
python3 process_data.py --data_dir [PSEUDO_LABEL_DIR] --save_name [SAVE_NAME]
```

PSEUDO_LABEL_DIR: auto-generated from get_pseduolabel_crohd and is set to "data" by default. 

SAVE_NAME: name of the aggregated data pickle file, default is all_crohd_data.pkl

We also release our generated all_crohd_data.pkl here:
https://drive.google.com/file/d/1sNNnQ37uIIcLVK3u_pNVpiwdF6rornYe/view?usp=drive_link

### 2. Fine-tune the pretrained motion model
Finetune the pretrained motion model from all generated pseudo-labels
```
python3 finetune_crohd.py --dataset_path [CROHD_DATASET_DIR] --pseudo_labels_path [AGGREGATED_DATA_PATH]
```
AGGREGATED_DATA_PATH: name of the aggregated data pickle file, default is all_crohd_data.pkl

Validation will be conducted every 500 iterations, and the first validation happening at the 0th iteration basically indicates how the pretrained baseline model performs.

In script, there is a configuration parameter named REQ_OCC. It means to use occlusion for more challenging tracking scenarios. Turning this on or off corresponds to two different columns in the Crohd table we include in the paper.

We also release our finetuned model at the 3000th iterations here:
https://drive.google.com/file/d/15f2rRSocI7kHzY_Desv5UfyU54fBZENn/view?usp=drive_link
