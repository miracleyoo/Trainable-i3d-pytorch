import argparse

parser = argparse.ArgumentParser('Runs inflated inception v1 network on\
    cricket sample from tensorflow demo (generate the network weights with\
    i3d_tf_to_pt.py first)')

# RGB arguments
parser.add_argument(
    '--rgb', action='store_true', help='Evaluate RGB pretrained network')

parser.add_argument(
    '--rgb_weights_path',
    type=str,
    default='model/model_rgb.pth',
    help='Path to rgb model state_dict')

parser.add_argument(
    '--rgb_sample_path',
    type=str,
    default='data/kinetic-samples/v_CricketShot_g04_c01_rgb.npy',
    help='Path to kinetics rgb numpy sample')

# Flow arguments
parser.add_argument(
    '--flow', action='store_true', help='Evaluate flow pretrained network')
parser.add_argument(
    '--flow_weights_path',
    type=str,
    default='model/model_flow.pth',
    help='Path to flow model state_dict')
parser.add_argument(
    '--flow_sample_path',
    type=str,
    default='data/kinetic-samples/v_CricketShot_g04_c01_flow.npy',
    help='Path to kinetics flow numpy sample')

parser.add_argument(
    '--classes_path',
    type=str,
    default='data/classes.txt',
    help='Path of the file containing classes names')

parser.add_argument(
    '--sample_path',
    type=str,
    help='Path of the sample video or image set you want to use to predict.')

parser.add_argument(
    '--sample_rate',
    type=int,
    default='1',
    help='Sample rate, or 1/sample_rate frames will be chosen.')

parser.add_argument(
    '--top_k',
    type=int,
    default='3',
    help='When display_samples, number of top classes to display')
