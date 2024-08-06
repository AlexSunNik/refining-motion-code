### 0. Download dataset
Download the horse10 dataset here:
https://drive.google.com/file/d/1XiKyJYEfnznnTppIB-CFNNvP99BRRr5n/view?usp=drive_link

We transform and reformat the labels from the original dataset into a pickle file for ease of use.

### 1. Pseudo-label generation
Start tracking clips loaded from the horse10 dataset for points in a window centered around interested query points (i.e. points you want to track). Trajectories that meet the Cycle Consistency check will be saved and kept, as described in the paper.

Run:
```
python3 get_pseduolabel_horse10.py --horse10_path [HORSE10_DATASET_DIR]
```

Visualizations are also logged during this process for you to perform sanity check on the generated pseudo-labels.

### 1.5 Aggregate and process generated data
Aggregate the pseudo-labels generated from all videos in the above step. Moreover, it will perform a speed filtering to only keep "useful" trajectories and discard those stationary trajectories which could be considered "too easy".

```
python3 process_data.py --data_dir [PSEUDO_LABEL_DIR] --save_name [SAVE_NAME]
```

PSEUDO_LABEL_DIR: auto-generated from get_pseduolabel_horse10 and is set to "data{THRESHOLD}" (e.g. data3). 

SAVE_NAME: name of the aggregated data pickle file, default is all_horse_data.pkl

We also release our generated all_horse_data.pkl here:
https://drive.google.com/file/d/1CUSBc2vAhO_GuthLuAiH5dPC5I64Nfyp/view?usp=drive_link

### 2. Fine-tune the pretrained motion model
Finetune the pretrained motion model from all generated pseudo-labels
```
python3 finetune_horse10.py --horse10_path [HORSE10_DATASET_DIR] --pseudo_labels_path [AGGREGATED_DATA_PATH]
```
AGGREGATED_DATA_PATH: name of the aggregated data pickle file, default is all_horse_data.pkl

Validation will be conducted every 500 iterations, and the first validation happening at the 0th iteration basically indicates how the pretrained baseline model performs.

We also release our finetuned model at the 3500th iterations here:
https://drive.google.com/file/d/1gxsrtzT7HQoaUmHAAbywb5ugFI9XpaTg/view?usp=drive_link

### 3. Evaluate baseline or finetuned model (Optional)
Above finetuning script periodically perform validation. 

However, to perform standalone evaluation of finetuned/pretrained models.
```
python3 eval_pips_horse10.py --horse10_path [HORSE10_DATASET_DIR] --checkpoint_path [CHECKPOINT_TO_EVAL]
```
Example CHECKPOINT_TO_EVAL for finetuned model:
'../reference_model/horsemodel_iter3500.pt'

Example CHECKPOINT_TO_EVAL for pretrained model:
'../reference_model/model-000200000.pth'