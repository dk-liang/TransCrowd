import argparse

parser = argparse.ArgumentParser(description='TransCrowd')

# Data specifications
parser.add_argument('--dataset', type=str, default='ShanghaiA',
                    help='choice train dataset')

parser.add_argument('--save_path', type=str, default='./save_file/A_baseline',
                    help='save checkpoint directory')

parser.add_argument('--workers', type=int, default=16,
                    help='load data workers')
parser.add_argument('--print_freq', type=int, default=200,
                    help='print frequency')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='start epoch for training')

# Model specifications
parser.add_argument('--test_dataset', type=str, default='UCF_QNRF',
                    help='choice train dataset')
parser.add_argument('--pre', type=str, default=None,
                    help='pre-trained model directory')
# parser.add_argument('--pre', type=str, default='./save_file/A_baseline_4/model_best_66.1.pth',
#                     help='pre-trained model directory')


# Optimization specifications
parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training')
parser.add_argument('--weight_decay', type=float, default=5 * 1e-4,
                    help='weight decay')
parser.add_argument('--momentum', type=float, default=0.95,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=20000,
                    help='number of epochs to train')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--best_pred', type=int, default=1e5,
                    help='best pred')
parser.add_argument('--gpu_id', type=str, default='0',
                    help='gpu id')

# nni config
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--model_type', type=str, default='token',
                    help='model type')

args = parser.parse_args()
return_args = parser.parse_args()
