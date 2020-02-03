import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Semantic Segmentation")

parser.add_argument('dataset', type=str, choices=['VOCAug', 'VOC2012', 'COCO', 'Cityscapes', 'ApolloScape', 'CULane'])
parser.add_argument('method', type=str, choices=['FCN', 'DeepLab', 'DeepLab3', 'PSPNet', 'ERFNet'])
parser.add_argument('train_list', type=str)
parser.add_argument('val_list', type=str)

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--dropout', '--do', default=0.1, type=float, metavar='DO', help='dropout ratio (default: 0.1)')
parser.add_argument('--train_size', default=840, type=int, metavar='L', help='size of training patches (default: 473)')
parser.add_argument('--test_size', default=840, type=int, metavar='L', help='size of testing patches (default: 513)')
parser.add_argument('--img_height', default=208, type=int, metavar='L', help='height of input images (default: 208)')
parser.add_argument('--img_width', default=976, type=int, metavar='L', help='width of input images (default: 976)')
parser.add_argument('--local_rank', type=int, default=0) # distributed data parallel

# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=24, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 20], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')

# ========================= Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--eval-freq', '-ef', default=1, type=int, metavar='N', help='evaluation frequency (default: 5)')

# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N', help='number of data loading workers (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--weight', default='', type=str, metavar='PATH', help='path to initial weight (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set') # true
parser.add_argument('--snapshot_pref', type=str, default="")
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
