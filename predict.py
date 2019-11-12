import torch
import numpy as np
from pre_process import pre_process
from src.i3dpt import I3D
from utils.utils import *
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_scores(sample, model):
    sample_var = torch.autograd.Variable(torch.from_numpy(sample.transpose(3, 0, 1, 2)[np.newaxis, :])).to(device)
    _, out_logit = model(sample_var)
    return out_logit


def main():
    # Initialize the RGB and Flow I3D model
    # Since we need to load pre-trained data, here we set num_classes=400
    # And we will change num_classes in load_and_freeze_model later
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_rgb.load_state_dict(torch.load(args.rgb_weights_path))
    i3d_rgb.to(device)
    out_rgb_logit = get_scores(rgb_data, i3d_rgb)

    i3d_flow = I3D(num_classes=400, modality='flow')
    i3d_flow.load_state_dict(torch.load(args.flow_weights_path))
    i3d_flow.to(device)
    out_flow_logit = get_scores(flow_data, i3d_flow)

    out_logit = out_rgb_logit + out_flow_logit
    out_softmax = torch.nn.functional.softmax(out_logit, 1).data.cpu()
    top_val, top_idx = torch.sort(out_softmax, 1, descending=True)

    print('===== Final predictions ====')
    print('logits proba class '.format(args.top_k))
    for i in range(args.top_k):
        logit_score = out_logit[0, top_idx[0, i]].data.item()
        print('{:.6e} {:.6e} {}'.format(logit_score, top_val[0, i],
                                        class_names[top_idx[0, i]]))


if __name__ == "__main__":
    args = parser.parse_args()
    class_names = [i.strip() for i in open(args.classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
    data_dir = Path('data/videos/pre-processed')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.sample_path is not None:
        out_path = Path("data/temp")
        if not out_path.exists(): out_path.mkdir()
        rgb_data, flow_data = pre_process(args.sample_path, sample_rate=args.sample_rate, out_path=out_path)
    else:
        log("Please using `--sample_path='your/video/path'` specify a video file or image set folder path to start.")
        exit(0)

    main()
