import copy
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from src.i3dpt import Unit3Dpy
from opts import parser
from pathlib2 import Path
from utils import *
from src.i3dpt import I3D
from DataLoader import RGBFlowDataset


def train_model(rgb_model, flow_model, criterion, optimizers, schedulers, num_epochs=25):
    since = time.time()

    best_rgb_model_wts = copy.deepcopy(rgb_model.state_dict())
    best_flow_model_wts = copy.deepcopy(flow_model.state_dict())

    best_acc = 0.0

    for epoch in range(num_epochs):
        log('Epoch {}/{}'.format(epoch, num_epochs - 1))
        log('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                rgb_model.train()  # Set model to training mode
                flow_model.train()
            else:
                rgb_model.eval()  # Set model to evaluate mode
                flow_model.train()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for rgb_data, flow_data, labels in data_loaders[phase]:
                rgb_data = rgb_data.to(device)
                flow_data = flow_data.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Calculate the joint output of two model
                    _, out_rgb_logit = rgb_model(rgb_data)
                    _, out_flow_logit = flow_model(flow_data)
                    out_logit = out_rgb_logit + out_flow_logit
                    out_softmax = torch.nn.functional.softmax(out_logit, 1)

                    _, preds = torch.max(out_softmax.data.cpu(), 1)
                    loss = criterion(out_softmax.cpu(), labels.cpu())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        for optimizer in optimizers:
                            optimizer.step()

                # statistics
                running_loss += loss.item() * rgb_data.shape[0]
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                for scheduler in schedulers:
                    scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            log('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_rgb_model_wts = copy.deepcopy(rgb_model.state_dict())
                best_flow_model_wts = copy.deepcopy(flow_model.state_dict())

        print()

    time_elapsed = time.time() - since
    log('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    log('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    rgb_model.load_state_dict(best_rgb_model_wts)
    flow_model.load_state_dict(best_flow_model_wts)

    return rgb_model, flow_model


def load_and_freeze_model(model, weight_path, num_classes, num_freeze=15):
    model.load_state_dict(torch.load(weight_path))
    log("Pre-trained model {} loaded successfully!".format(weight_path))
    counter = 0
    for child in model.children():
        counter += 1
        if counter < num_freeze:
            log("Layer {} frozen!".format(child._get_name()))
            for param in child.parameters():
                param.requires_grad = False
    model.num_classes = num_classes
    model.conv3d_0c_1x1 = Unit3Dpy(in_channels=1024,
                                   out_channels=num_classes,
                                   kernel_size=(1, 1, 1),
                                   activation=None,
                                   use_bias=True,
                                   use_bn=False)


def get_scores(sample_var, model):
    out_var, out_logit = model(sample_var)
    out_tensor = out_var.data.cpu()

    top_val, top_idx = torch.sort(out_tensor, 1, descending=True)

    log('Top {} classes and associated probabilities: '.format(args.top_k))
    for i in range(args.top_k):
        print('[{}]: {:.6E}'.format(class_names[top_idx[0, i]],
                                    top_val[0, i]))
    return out_logit


if __name__ == "__main__":
    args = parser.parse_args()
    class_names = [i.strip() for i in open(args.classes_path)]
    class_dicts = {k: v for v, k in enumerate(class_names)}
    data_dir = Path('data/videos/pre-processed')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rgb_flow_datasets = {x: RGBFlowDataset(data_dir / x, class_dicts, sample_rate=args.sample_rate)
                         for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(rgb_flow_datasets[x], batch_size=1,
                                                   shuffle=True, num_workers=0)
                    for x in ['train', 'val']}
    dataset_sizes = {x: len(rgb_flow_datasets[x]) for x in ['train', 'val']}

    # Initialize the RGB and Flow I3D model
    # Since we need to load pre-trained data, here we set num_classes=400
    # And we will change num_classes in load_and_freeze_model later
    i3d_rgb = I3D(num_classes=400, modality='rgb')
    i3d_flow = I3D(num_classes=400, modality='flow')

    load_and_freeze_model(model=i3d_rgb, num_classes=len(class_names), weight_path=args.rgb_weights_path)
    load_and_freeze_model(model=i3d_flow, num_classes=len(class_names), weight_path=args.flow_weights_path)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_rgb = optim.SGD(filter(lambda p: p.requires_grad, i3d_rgb.parameters()), lr=0.001, momentum=0.9)
    optimizer_flow = optim.SGD(filter(lambda p: p.requires_grad, i3d_flow.parameters()), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler_rgb = lr_scheduler.StepLR(optimizer_rgb, step_size=7, gamma=0.1)
    exp_lr_scheduler_flow = lr_scheduler.StepLR(optimizer_flow, step_size=7, gamma=0.1)

    optimizers = (optimizer_rgb, optimizer_flow)
    schedulers = (exp_lr_scheduler_rgb, exp_lr_scheduler_flow)

    train_model(i3d_rgb, i3d_flow, criterion, optimizers, schedulers, num_epochs=25)
