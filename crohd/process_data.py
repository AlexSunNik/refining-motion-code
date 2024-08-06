import pickle
import os
import io
import torch
import argparse

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def main():
    parser = argparse.ArgumentParser()
    # pseduolabels directory
    parser.add_argument('--data_dir', type=str, default='./data/')
    # saved aggregated data pkl name
    parser.add_argument('--save_name', type=str, default='all_crohd_data.pkl')
    args = parser.parse_args()
    directory = args.data_dir
    all_data = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, 'rb') as handle:
            final_data = CPU_Unpickler(handle).load()
            all_data += final_data

    save_name = args.save_name
    with open(save_name, 'wb') as handle:
        pickle.dump(all_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()