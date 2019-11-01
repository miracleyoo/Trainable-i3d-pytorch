import time

from pathlib2 import Path


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


def build_data_path(is_image=False):
    if is_image:
        data_root = Path("data/images/")
    else:
        data_root = Path("data/videos/")
    raw_path = data_root / "raw"
    train_path = data_root / "pre-processed"
    if not train_path.exists(): train_path.mkdir()
    first_parts = ["train", "val"]
    class_parts = [i.stem for i in raw_path.iterdir() if not i.stem.startswith(".") and i.is_dir()]
    for first_part in first_parts:
        temp = train_path/first_part
        if not temp.exists(): temp.mkdir()
        for class_part in class_parts:
            # print(str(train_path/first_part/class_part))
            temp = train_path / first_part / class_part
            if not temp.exists(): temp.mkdir()
