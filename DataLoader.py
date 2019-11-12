import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import Dataset
from utils.temporal_transforms import *
from PIL import Image

import torchvision.transforms.functional as TF
import torchvision.transforms as transforms


class SpacialTransform(Dataset):
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size

    def refresh_random(self, image_np):
        # Random crop
        self.crop_dim = transforms.RandomCrop.get_params(
            Image.fromarray(image_np), output_size=self.output_size)

        # Random horizontal flipping
        self.horizontal_flip = False
        if random.random() > 0.5:
            self.horizontal_flip = True

        # Random vertical flipping
        self.vertical_flip = False
        if random.random() > 0.5:
            self.vertical_flip = True

    def transform(self, image_nps):
        image_PILs = []
        # Resize
        # resize = transforms.Resize(size=(520, 520))
        # image_left = resize(image_left)
        # image_right = resize(image_right)

        for image_np in image_nps:
            image_PIL = Image.fromarray(image_np)
            image_PIL = TF.crop(image_PIL, *self.crop_dim)
            if self.horizontal_flip: image_PIL = TF.hflip(image_PIL)
            if self.vertical_flip: image_PIL = TF.vflip(image_PIL)
            image_PIL = TF.to_tensor(image_PIL)
            image_PILs.append(image_PIL)
        return torch.stack(image_PILs)


class RGBFlowDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, class_dict, sample_rate=1, sample_type="num", fps=5, out_frame_num=32, augment=False):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        self.root_dir = Path(root_dir)
        self.sub_dirs = [i for i in self.root_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
        self.class_names = [i.stem for i in self.sub_dirs]
        self.data_pairs = []
        self.spacial_transform = SpacialTransform()
        self.temporal_transform = TemporalRandomCrop(out_frame_num)
        # self.temporal_transform =
        for sub_dir in self.sub_dirs:
            item_dirs = [i for i in sub_dir.iterdir() if i.is_dir() and not i.stem.startswith('.')]
            for item_dir in item_dirs:
                contents = [i for i in item_dir.iterdir() if i.is_file() and not i.stem.startswith('.')]
                if contents:
                    # print(contents, sample_rate)
                    try:
                        if sample_type == "num":
                            temp_rgb = [i for i in contents if i.stem.startswith('rgb') and "SampleRate" in i.stem
                                        and int(i.stem.split('_')[-1]) == sample_rate][0]
                            temp_flow = [i for i in contents if i.stem.startswith('flow') and "SampleRate" in i.stem
                                         and int(i.stem.split('_')[-1]) == sample_rate][0]
                        elif sample_type == "fps":
                            temp_rgb = [i for i in contents if i.stem.startswith('rgb') and "FPS" in i.stem
                                        and int(i.stem.split('_')[-1]) == fps][0]
                            temp_flow = [i for i in contents if i.stem.startswith('flow') and "FPS" in i.stem
                                         and int(i.stem.split('_')[-1]) == fps][0]
                        else:
                            raise ValueError("sample_type should be 'num' or 'fps'")
                    except IndexError:
                        raise IndexError("Please make sure you specified input sample num or fps right.")
                    self.data_pairs.append((temp_rgb, temp_flow, class_dict[sub_dir.stem]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        rgb_data = np.float32(np.load(self.data_pairs[idx][0]))
        self.spacial_transform.refresh_random(rgb_data[0])
        rgb_data = rgb_data.transpose(3, 0, 1, 2)
        rgb_data = self.temporal_transform(rgb_data)
        rgb_data = self.spacial_transform.transform(rgb_data)
        flow_data = np.float32(np.load(self.data_pairs[idx][1]).transpose(3, 0, 1, 2))
        flow_data = self.temporal_transform(flow_data)
        flow_data = self.spacial_transform.transform(flow_data)
        # print("Flow: ", flow_data.shape, "Rgb: ", rgb_data.shape, self.data_pairs[idx][0], self.data_pairs[idx][-1])
        return rgb_data, flow_data, self.data_pairs[idx][2]
