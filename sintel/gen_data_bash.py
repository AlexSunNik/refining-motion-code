scene_names = ['alley_1', 'alley_2', 'ambush_2', 'ambush_4', 'ambush_5', 'ambush_6', 'ambush_7', 'bamboo_1', \
'bamboo_2', 'bandage_1', 'bandage_2', 'cave_2', 'cave_4', 'market_2', 'market_5', 'market_6', \
'mountain_1', 'shaman_2', 'shaman_3', 'sleeping_1', 'sleeping_2', 'temple_2', 'temple_3']
import os

gpu = 0
total_gpus = 4
per_gpu = len(scene_names) // total_gpus
sh_name = "create_pseudo_tracks"
postfix = ".sh"
i = 0
while i < len(scene_names):
    gpu = i % 4
    scene_name = scene_names[i]
    file_name = f"{sh_name}_{gpu}_{postfix}"
    with open(file_name, 'a') as f:
        command = f"CUDA_VISIBLE_DEVICES={gpu} python3 cycle_consty_train_aug.py --scene_name {scene_name} --gen_data >> {scene_name}_gendata.txt\n"
        f.write(command)
    i += 1
# sh_name = "create_pseudo_tracks.sh"

# gpu = 0
# total_gpus = 4
# with open(sh_name, 'w') as f:
#     for video_name in train_video_names:
#         if video_name in ["roundabout", "cows"]:
#             continue
#         if gpu % 4 == 3:
#             command = f"CUDA_VISIBLE_DEVICES={gpu % 4} python3 gen_tapdavis_cycleconsty_single_vid.py --video_name {video_name} \n"
#         else:
#             command = f"CUDA_VISIBLE_DEVICES={gpu % 4} python3 gen_tapdavis_cycleconsty_single_vid.py --video_name {video_name} & \n"
#         gpu += 1
#         f.write(command)
