import argparse
import random

import cv2
import numpy as np

from utils import *

_VIDEO_EXT = ['.avi', '.mp4', '.mov']
_IMAGE_EXT = ['.jpg', '.png']
_IMAGE_SIZE = 224


def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(n))]


class FrameGenerator(object):
    def __init__(self, input_path, sample_num=1, random_choice=False, chosen_frames=None):
        """
        :param input_path: The input video file or image set path
        :param sample_num: The number of frames you hope to use, they are chosen evenly spaced
        :param slice_num: The number of blocks you want to divide the input file into, and frames
                            are randomly chosen from each block.
        """
        input_path = Path(input_path)
        self.is_video = input_path.is_file() and input_path.suffix.lower() in _VIDEO_EXT
        if self.is_video:
            self.video_object = cv2.VideoCapture(str(input_path))
            self.frame_num = self.get_frame_num()
        elif input_path.is_dir():
            self.image_list = [i for i in input_path.iterdir() if not
            i.stem.startswith('.') and i.suffix.lower() in _IMAGE_EXT]
            self.image_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem))))
            self.frame_num = len(self.image_list)
        else:
            raise IOError("Input data path is not valid! Please make sure it is whether "
                          "a video file or a image set directory")
        self.counter = 0
        self.current_video_frame = -1
        self.sample_num = sample_num
        # self.skip_num = int(np.ceil(self.frame_num / self.sample_num))
        # range(self.frame_num-1): Random choice shouldn't chooses the last frame, since flow need the next frame
        # self.full_frame_lists = [range(i, min(i + self.skip_num, self.frame_num))
        #                          for i in [j for j in range(self.frame_num-1) if j % self.skip_num == 0]]
        self.full_frame_lists = split(list(range(self.frame_num-1)), self.sample_num)

        if chosen_frames is None:
            if random_choice:
                self.chosen_frames = [random.choice(i) for i in self.full_frame_lists]
            else:
                self.chosen_frames = [i[0] for i in self.full_frame_lists]
        else:
            self.chosen_frames = chosen_frames

    def get_frame_num(self):
        if cv2.__version__ >= '3.0.0':
            cap_prop_frame_count = cv2.CAP_PROP_FRAME_COUNT
        else:
            cap_prop_frame_count = cv2.cv.CV_CAP_PROP_FRAME_COUNT
        frame_num = int(self.video_object.get(cap_prop_frame_count))
        return frame_num

    def __len__(self):
        return len(self.chosen_frames)

    def release(self):
        if self.is_video:
            self.video_object.release()

    def get_frame(self):
        if self.is_video:
            # print(self.current_video_frame, self.chosen_frames[self.counter])
            while True:
                _, frame = self.video_object.read()
                self.current_video_frame += 1
                if self.current_video_frame-1 != self.chosen_frames[self.counter]:
                    break
        else:
            frame = cv2.imread(str(self.image_list[self.chosen_frames[self.counter]]), 3)
        frame = cv2.resize(frame, (_IMAGE_SIZE, _IMAGE_SIZE))
        self.counter += 1
        return frame

    def get_next_frame(self):
        if self.is_video:
            _, frame = self.video_object.read()
            self.current_video_frame += 1
        else:
            frame = cv2.imread(str(self.image_list[self.chosen_frames[self.counter-1]+1]), 3)
        frame = cv2.resize(frame, (_IMAGE_SIZE, _IMAGE_SIZE))
        return frame


def compute_rgb(video_path, sample_rate=1, out_path=None, random_choice=False, chosen_frames=None):
    """Compute RGB"""
    rgb = []
    if out_path is None:
        out_path = Path(*video_path.parts[:-3], "pre-processed", "train",
                        video_path.parts[-2], video_path.stem)
        if not out_path.exists(): out_path.mkdir()
    else:
        out_path = Path(out_path)
    out_path = out_path / ('rgb-SampleRate_{}.npy'.format(sample_rate))

    video_object = FrameGenerator(video_path, sample_rate, random_choice, chosen_frames)
    for i in range(len(video_object)):
        frame = video_object.get_frame()
        frame = (frame / 255.)  # * 2 - 1
        rgb.append(frame)
    video_object.release()
    # rgb = rgb[:-1]
    rgb = np.float32(np.array(rgb))
    log('save rgb with shape ', rgb.shape)
    np.save(out_path, rgb)
    return rgb, video_object.chosen_frames


def compute_flow(video_path, sample_rate=1, out_path=None, random_choice=False, chosen_frames=None):
    """Compute the TV-L1 optical flow."""
    flow = []
    if out_path is None:
        out_path = Path(*video_path.parts[:-3], "pre-processed", "train",
                        video_path.parts[-2], video_path.stem)
        if not out_path.exists(): out_path.mkdir()
    else:
        out_path = Path(out_path)
    out_path = out_path / ('flow-SampleRate_{}.npy'.format(sample_rate))

    bins = np.linspace(-20, 20, num=256)
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    video_object = FrameGenerator(video_path, sample_rate, random_choice, chosen_frames)
    # curr = None
    for i in range(len(video_object)):
        # if sample_rate == 1 and curr is not None:
        #     prev = curr
        # else:
        frame1 = video_object.get_frame()
        prev = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

        frame2 = video_object.get_next_frame()
        curr = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        curr_flow = TVL1.calc(prev, curr, None)
        assert (curr_flow.dtype == np.float32)

        # Truncate large motions
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20

        # digitize and scale to [-1;1]
        curr_flow = np.digitize(curr_flow, bins)
        curr_flow = (curr_flow / 255.) * 2 - 1

        # Append this flow frame
        flow.append(curr_flow)

    video_object.release()
    flow = np.float32(np.array(flow))
    log('Save flow with shape ', flow.shape)

    np.save(out_path, flow)
    return flow, video_object.chosen_frames


def pre_process(video_path, sample_rate=1, out_path=None, random_choice=False):
    video_path = Path(video_path)
    with Timer('Compute RGB'):
        log('Extract RGB...')
        rgb_data, chosen_frames = compute_rgb(video_path, sample_rate, out_path, random_choice)

    with Timer('Compute flow'):
        log('Extract Flow...')
        flow_data = compute_flow(video_path, sample_rate, out_path, random_choice, chosen_frames)
    return rgb_data, flow_data


def mass_process(is_image=False, sample_rate=1, random_choice=False):
    if is_image:
        data_root = Path("data/images/")
    else:
        data_root = Path("data/videos/")
    raw_path = data_root / "raw"
    class_paths = [i for i in raw_path.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    item_paths = []
    for class_path in class_paths:
        if is_image:
            item_paths.extend([i for i in class_path.iterdir()
                               if not i.stem.startswith(".") and i.is_dir()])
        else:
            item_paths.extend([i for i in class_path.iterdir()
                               if not i.stem.startswith(".") and i.is_file() and i.suffix.lower() in _VIDEO_EXT])
    for item_path in item_paths:
        with Timer(item_path.name):
            log("Now start processing:", str(item_path), "Sample rate:", sample_rate)
            pre_process(item_path, sample_rate=sample_rate, random_choice=random_choice)


def main(video_path, sample_rate, random_choice=False):
    pre_process(video_path, sample_rate=sample_rate, random_choice=random_choice)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Pre-process the video into formats which i3d uses.')

    # RGB arguments
    parser.add_argument('--use_image', action='store_true',
                        help='Use a series of image(its folder) as video input')
    parser.add_argument('--random_choice', action='store_true',
                        help='Whether to choose frames randomly or uniformly')
    parser.add_argument('--mass', action='store_true',
                        help='Compute RGBs and Flows massively.')
    parser.add_argument('--init_dir', action='store_true',
                        help='Initialize the data pre-processed folder tree.')
    parser.add_argument(
        '--input_path',
        type=str,
        default='data/videos/raw/take-out/IMG_6801.mp4',
        help='Path to input video or images folder')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default='1',
        help='Sample rate, or 1/sample_rate frames will be chosen.')
    args = parser.parse_args()
    if args.use_image:
        DATA_ROOT = Path('data/images/')
    else:
        DATA_ROOT = Path('data/videos/')
    DATA_DIR = DATA_ROOT / 'raw'
    SAVE_DIR = DATA_ROOT / 'pre-processed'
    if args.init_dir:
        build_data_path(args.use_image)
    if args.mass:
        mass_process(args.use_image, args.sample_rate, args.random_choice)
    else:
        main(args.input_path, args.sample_rate, args.random_choice)
