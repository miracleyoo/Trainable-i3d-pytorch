import argparse
import shutil
import time

from pathlib2 import Path
from sklearn.model_selection import train_test_split


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.t_start = time.time()

    def __exit__(self, type, value, traceback):
        print("==> [%s]:\t" % self.name, end="")
        self.time_elapsed = time.time() - self.t_start
        print("Elapsed Time: %s (s)" % self.time_elapsed)


def log(*snippets, end=None):
    if end is None:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]))
    else:
        print(time.strftime("==> [%Y-%m-%d %H:%M:%S]", time.localtime()) + " " + "".join([str(s) for s in snippets]),
              end=end)


def build_data_path(is_image=False, data_root=None):
    if data_root is None:
        if is_image:
            data_root = Path("../data/images/")
        else:
            data_root = Path("../data/videos/")
    raw_path = data_root / "raw"
    train_path = data_root / "pre-processed"
    if not train_path.exists(): train_path.mkdir()
    first_parts = ["train", "val"]
    class_parts = [i.stem for i in raw_path.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    for first_part in first_parts:
        temp_1 = train_path/first_part
        if not temp_1.exists(): temp_1.mkdir()
        for class_part in class_parts:
            # print(str(train_path/first_part/class_part))
            temp_2 = train_path / first_part / class_part
            if not temp_2.exists(): temp_2.mkdir()


def sep_train_val(is_image=False, train_path=None, val_ratio=0.2):
    if train_path is None:
        if is_image:
            train_path = "../data/images/pre-processed/train"
        else:
            train_path = "../data/videos/pre-processed/train"
    train_path = Path(train_path)
    class_parts = [i for i in train_path.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    for class_part in class_parts:
        data_temp = [i for i in class_part.iterdir() if not i.stem.startswith(".")]
        data_temp_train, data_temp_val = train_test_split(data_temp, test_size=val_ratio, random_state=9854)
        for data_path in data_temp_val:
            new_path = Path(*data_path.parts[:-3], "val", *data_path.parts[-2:])
            log(str(data_path), "-->", str(new_path))
            shutil.move(str(data_path), str(new_path))


def coalesce_train_val(is_image=False, val_path=None):
    if val_path is None:
        if is_image:
            val_path = "../data/images/pre-processed/val"
        else:
            val_path = "../data/videos/pre-processed/val"
    val_path = Path(val_path)
    class_parts = [i for i in val_path.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    for class_part in class_parts:
        data_temp = [i for i in class_part.iterdir() if not i.stem.startswith(".")]
        for data_path in data_temp:
            new_path = Path(*data_path.parts[:-3], "train", *data_path.parts[-2:])
            log(str(data_path), "-->", str(new_path))
            shutil.move(str(data_path), str(new_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Some utils.')

    # RGB arguments
    parser.add_argument('--sep', action='store_true',
                        help='Separate the full data into train and val, and save them in corresponding folders')
    parser.add_argument('--coa', action='store_true',
                        help='Coalesce train and val data separated all to train data folder.')
    parser.add_argument('--bdp', action='store_true', help='Build data directory path. You need to make sure you'
                                                           ' have placed your data in raw folder in advance correctly')
    parser.add_argument('--use_image', action='store_true',
                        help='Use a series of image(its folder) as video input')

    parser.add_argument(
        '--input_path',
        type=str,
        # default='data/videos/pre-processed/train',
        help='Path to video or images folder')
    parser.add_argument(
        '--val_ratio',
        type=float,
        default='0.2',
        help='The ratio of val data.')
    args = parser.parse_args()
    if args.sep:
        sep_train_val(is_image=args.use_image, train_path=args.input_path, val_ratio=args.val_ratio)
    elif args.coa:
        coalesce_train_val(is_image=args.use_image, val_path=args.input_path)
    elif args.bdp:
        build_data_path(is_image=args.use_image)
