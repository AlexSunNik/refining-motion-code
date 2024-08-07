### 0. Download dataset
Download the sintel dataset following the instructions here:
http://sintel.is.tue.mpg.de/


### 1. Pseudo-label generation
Start tracking flows loaded from the sintel dataset. Trajectories that meet the Cycle consistency and Color constancy criteria will be saved and kept, as described in the paper.

Run:
```
python3 refine_raft_sintel.py --scene_name [SINTEL_SCENE_NAME] --split [SINTEL_SPLIT] --sintel_path [SINTEL_LOCATION] --gen_data
```
SINTEL_SCENE_NAME: which scene from 23 Sintel scenes to use for gathering pseudo-labels. E.g. alley_1, bandage_1, etc.

SINTEL_SPLIT: choose between [final, clean]

SINTEL_LOCATION: where is the downloaded sintel dataset

--gen_data: flag to only generate flows. without this flag, refine_raft_sintel.py will directly start finetuning on RAFT using the generated flows.

We also provide bash script to generate data for all scenes together. Run:
```
bash get_pseudolabel_sintel.sh
```
You can modify the bash script with non-blocking commands to run them parallely.

By default, script will create a folder named "raft_pseudo_labels" and dump the collected pseudo labels there.

### 2. Fine-tune the pretrained motion model
Finetune the pretrained motion model from all generated pseudo-labels

Run:
```
python3 refine_raft_sintel.py --scene_name [SINTEL_SCENE_NAME] --split [SINTEL_SPLIT] --sintel_path [SINTEL_LOCATION]
```

Validation will be conducted every 100 iterations, and the first validation happening at the 0th iteration basically indicates how the pretrained baseline model performs.

As above, we provide a bash script to train for all scenes together. Run:
```
bash finetune_sintel.sh
```
You can modify the bash script with non-blocking commands to run them parallely.

Note that you can essenstially skip the first step and directly run this, the script will first generate pseudo-labels then immediately start finetuning with them.