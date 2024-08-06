# run get_pseudolabel_crohd.py parallely for all dset_split
# parallel option
# CUDA_VISIBLE_DEVICES=0 python3 get_pseudolabel_crohd.py --dset_split 0 &
# CUDA_VISIBLE_DEVICES=1 python3 get_pseudolabel_crohd.py --dset_split 1 &
# CUDA_VISIBLE_DEVICES=2 python3 get_pseudolabel_crohd.py --dset_split 2 &
# CUDA_VISIBLE_DEVICES=3 python3 get_pseudolabel_crohd.py --dset_split 3 &
# CUDA_VISIBLE_DEVICES=4 python3 get_pseudolabel_crohd.py --dset_split 4 &
# CUDA_VISIBLE_DEVICES=5 python3 get_pseudolabel_crohd.py --dset_split 5

# non-parallel option
python3 get_pseudolabel_crohd.py --dset_split 0
python3 get_pseudolabel_crohd.py --dset_split 1
python3 get_pseudolabel_crohd.py --dset_split 2
python3 get_pseudolabel_crohd.py --dset_split 3
python3 get_pseudolabel_crohd.py --dset_split 4
python3 get_pseudolabel_crohd.py --dset_split 5