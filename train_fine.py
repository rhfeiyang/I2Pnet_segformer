import os
import argparse
import random
import numpy as np
import importlib.util
import torch.backends.cudnn as cudnn

import torch
from isegm.utils.exp import init_experiment, str2bool


def main():
    args = parse_args()
    if args.temp_model_path:
        model_script = load_module(args.temp_model_path)
    else:
        model_script = load_module(args.model_path)

    model_base_name = getattr(model_script, 'MODEL_NAME', None)

    args.distributed = 'WORLD_SIZE' in os.environ
    cfg = init_experiment(args, model_base_name)

    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_sharing_strategy('file_system')

    model_script.main(cfg)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='Path to the model script.')

    parser.add_argument('--exp-name', type=str, default='',
                        help='Here you can specify the name of the experiment. '
                             'It will be added as a suffix to the experiment folder.')

    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='Dataloader threads.')

    parser.add_argument('--batch-size', type=int, default=-1,
                        help='You can override model batch size by specify positive number.')

    parser.add_argument('--ngpus', type=int, default=1,
                        help='Number of GPUs. '
                             'If you only specify "--gpus" argument, the ngpus value will be calculated automatically. '
                             'You should use either this argument or "--gpus".')

    parser.add_argument('--gpus', type=str, default='', required=False,
                        help='Ids of used GPUs. You should use either this argument or "--ngpus".')

    parser.add_argument('--resume-exp', type=str, default=None,
                        help='The prefix of the name of the experiment to be continued. '
                             'If you use this field, you must specify the "--resume-prefix" argument.')

    parser.add_argument('--resume-prefix', type=str, default='latest',
                        help='The prefix of the name of the checkpoint to be loaded.')

    parser.add_argument('--start-epoch', type=int, default=0,
                        help='The number of the starting epoch from which training will continue. '
                             '(it is important for correct logging and learning rate)')

    parser.add_argument('--weights', type=str, default=None,
                        help='Model weights will be loaded from the specified path if you use this argument.')

    parser.add_argument('--temp-model-path', type=str, default='',
                        help='Do not use this argument (for internal purposes).')

    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--att", type=str2bool, default=False)

    parser.add_argument("--fbi", action='store_true', default=True, help='Use Foreground and Background Attention')

    parser.add_argument('--int-net-mode', dest='int_net_mode', help='user intention network mode', default='att+upatt')

    parser.add_argument('--int-model', dest='int_model_name', required=True, help='user intention network checkpoint name')

    parser.add_argument('--seg-net-mode', dest='seg_net_mode', help='segmentation network mode', default='att+upatt')

    args = parser.parse_args()

    args.seg_model_name = args.int_model_name

    return args


def load_module(script_path):
    spec = importlib.util.spec_from_file_location("model_script", script_path)
    model_script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_script)

    return model_script


if __name__ == '__main__':
    main()