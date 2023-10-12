
import pickle
from pathlib import Path

if __name__ == '__main__':
    action_space = len(list(pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb')).keys()))
    state_size = 9 + action_space * 4

    print('State size:', state_size)
    print('Action space:', action_space)
