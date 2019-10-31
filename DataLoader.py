import numpy as np
from pathlib2 import Path
from torch.utils.data import Dataset


class RGBFlowDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, class_dict, sample_rate=1):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in self.root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.data_pairs = []
        for sub_dir in self.sub_dirs:
            item_dirs = [i for i in sub_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
            for item_dir in item_dirs:
                contents = [i for i in item_dir.iterdir() if i.is_file() and not i.stem.startswith('.')]
                if contents:
                    temp_rgb = [i for i in contents if i.stem.startswith('rgb')
                                and int(i.stem.split('_')[-1]) == sample_rate][0]
                    temp_flow = [i for i in contents if i.stem.startswith('flow')
                                 and int(i.stem.split('_')[-1]) == sample_rate][0]
                    self.data_pairs.append((temp_rgb, temp_flow, class_dict[sub_dir.stem]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        rgb_data = np.load(self.data_pairs[idx][0]).transpose(3, 0, 1, 2)
        flow_data = np.load(self.data_pairs[idx][1]).transpose(3, 0, 1, 2)
        return np.float32(rgb_data), np.float32(flow_data), self.data_pairs[idx][2]
