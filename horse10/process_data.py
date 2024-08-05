
import pickle
import os
import io
import torch
import argparse

SPEED_THRESH = 5
TRAIN_N = 128

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def compute_motion(x):
    total_dist = None
    for i in range(1, 8):
        dist = torch.linalg.norm(x[i] - x[i-1], dim=1)
        total_dist = dist if total_dist is None else total_dist + dist
    speed = total_dist / 8
    return speed

def main():
    parser = argparse.ArgumentParser()
    # pseduolabels directory
    parser.add_argument('--data_dir', type=str, default='./data3/')
    # saved aggregated data pkl name
    parser.add_argument('--save_name', type=str, default='all_horse_data.pkl')
    args = parser.parse_args()

    with open("idx2name_map.pkl", 'rb') as handle:
        idx2name_map = CPU_Unpickler(handle).load()


    summarized_data = {}
    directory = args.data_dir
    all_data = []
    start_to_idxs_map = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, 'rb') as handle:
            final_data = CPU_Unpickler(handle).load()
            for i in range(len(final_data)):
                final_data[i] = (idx2name_map[final_data[i][0]], final_data[i][1], final_data[i][2])
                video_name = final_data[i][0]
                idxs = final_data[i][1]
                idxs = [int(x[0]) for x in idxs]
                start_to_idxs_map[idxs[0]] = idxs
                key = (video_name, idxs[0])
                if key not in summarized_data:
                    summarized_data[key] = []
                summarized_data[key].append(final_data[i][2])

    for key in summarized_data:
        summarized_data[key] = torch.cat(summarized_data[key], 1)

    all_speed = []
    for x in summarized_data:
        v = summarized_data[x]
        print(x, v.shape)
        speed = compute_motion(v)
        speed_idxs = speed > SPEED_THRESH
        summarized_data[x] = summarized_data[x][:, speed_idxs]

    all_items = []
    for key, val in summarized_data.items():
        video_name, idx_start = key
        all_idxs = start_to_idxs_map[idx_start]

        cur_N = val.shape[1]
        start = 0
        while True:
            end = start + TRAIN_N
            if end > cur_N:
                break
            all_items.append((video_name, all_idxs, val[:, start:end]))
            start = end

    save_name = args.save_name
    with open(save_name, 'wb') as handle:
        pickle.dump(all_items, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
