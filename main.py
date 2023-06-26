import sys
import os
import math
import argparse
import pickle
import torch
import importlib

from tqdm import tqdm
from agd  import AGD

from architecture.fcn    import *
from architecture.vgg    import *
from architecture.resnet import *

############################################################################################
######################################### Parse args #######################################
############################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--arch',       type=str,   default='fcn',      choices=['fcn', 'vgg', 'resnet18', 'resnet50']       )
parser.add_argument('--dataset',    type=str,   default='cifar10',  choices=['cifar10', 'cifar100', 'mnist', 'imagenet'] )
parser.add_argument('--loss',       type=str,   default='mse',      choices=['mse', 'xent']                              )
parser.add_argument('--train_bs',   type=int,   default=128  )
parser.add_argument('--test_bs',    type=int,   default=128  )
parser.add_argument('--epochs',     type=int,   default=200  )
parser.add_argument('--depth',      type=int,   default=10   )
parser.add_argument('--width',      type=int,   default=256  )
parser.add_argument('--distribute', action='store_true'      )
parser.add_argument('--cpu',        action='store_true'      )
parser.add_argument('--gain',       type=float, default=1.0  )
args = parser.parse_args()

############################################################################################
#################################### Distributed setup #####################################
############################################################################################

local_rank = 0

if args.distribute:
    world_size  = int(os.getenv('OMPI_COMM_WORLD_SIZE'))
    global_rank = int(os.getenv('OMPI_COMM_WORLD_RANK'))
    local_rank  = global_rank % torch.cuda.device_count()

    torch.distributed.init_process_group(backend='nccl', rank=global_rank, world_size=world_size)
    print(f'GPU {global_rank} reporting in. Local rank: {local_rank}. CPU threads: {torch.get_num_threads()}.')
    torch.distributed.barrier()

    if global_rank > 0:
        tqdm = lambda x, total : x
        sys.stdout = open(os.devnull, 'w')

############################################################################################
####################################### Print args #########################################
############################################################################################

print("{: <39} {: <20}".format("\nArgument", "Value"))
print("{: <39} {: <20}".format(*["=============================="]*2))
for arg in vars(args):
    print("{: <39} {: <20}".format(arg, getattr(args, arg)))
print("\nNote: depth and width are only used for fully-connected networks.")

############################################################################################
######################################### Get data #########################################
############################################################################################

print("\nGetting data...")
print("==================================="*2)

data_module = importlib.import_module("data."+args.dataset)
trainset, testset, input_dim, output_dim = data_module.getData()

if args.distribute:
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
    train_loader = torch.utils.data.DataLoader( trainset,
                                                batch_size=int(args.train_bs/world_size),
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                sampler=train_sampler   )
    test_loader = torch.utils.data.DataLoader(  testset,
                                                batch_size=int(args.test_bs/world_size),
                                                shuffle=False,
                                                num_workers=8,
                                                pin_memory=True,
                                                sampler=test_sampler    )
else:
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.train_bs, shuffle=True,  pin_memory=True)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=args.test_bs,  shuffle=False, pin_memory=True)

############################################################################################
##################################### Set architecture #####################################
############################################################################################

if args.arch == 'fcn':
    net = FCN(args.depth, args.width, input_dim, output_dim)
elif args.dataset == 'imagenet' and args.arch == 'resnet50':
    net = resnet50(num_classes=1000)
elif 'cifar' not in args.dataset:
    raise Exception("That network only works with CIFAR.")
elif args.arch == 'vgg':
    net = VGG16(output_dim)
elif args.arch == 'resnet18':
    net = PreActResNet18(output_dim)
elif args.arch == 'resnet50':
    net = PreActResNet50(output_dim)

if not args.cpu:
    net = net.cuda(local_rank)

agd = AGD(net, args.gain)
agd.init_weights()

if args.distribute:
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[local_rank])

print("{: <39} {: <20}".format("\nLayer", "Shape"))
print("{: <39} {: <20}".format(*["=============================="]*2))
for name, p in net.named_parameters():
    print("{: <39} {: <20}".format(name, str(list(p.shape))))

############################################################################################
######################################## Define loop #######################################
############################################################################################

def loop(net, dataloader, optim, train):
    net.train() if train else net.eval()

    num_minibatches = len(dataloader)
    
    epoch_loss = 0
    epoch_acc  = 0
    epoch_log  = 0

    for data, target in tqdm(dataloader, total=num_minibatches):
        if not args.cpu:
            data, target = data.cuda(local_rank), target.cuda(local_rank)
        output = net(data)

        if args.loss == 'mse':
            onehot = torch.nn.functional.one_hot(target, num_classes=output.shape[1]).float()
            onehot *= math.sqrt(output.shape[1])
            loss = (output-onehot).square().mean()
        elif args.loss == 'xent':
            error = - output[range(target.shape[0]),target] + output.logsumexp(dim=1)
            loss = error.mean()
        if train: loss.backward()

        acc = (output.argmax(dim=1) == target).float().mean()

        if args.distribute:
            torch.distributed.all_reduce(loss, torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acc,  torch.distributed.ReduceOp.SUM)

            loss /= world_size
            acc  /= world_size

        if train:
            epoch_log += optim.step()
            net.zero_grad()

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    return epoch_loss / num_minibatches, epoch_acc / num_minibatches, epoch_log / num_minibatches


############################################################################################
###################################### Train network #######################################
############################################################################################

results = {}
results['log_list'       ] = []
results['train_loss_list'] = []
results['test_loss_list' ] = []
results['train_acc_list' ] = []
results['test_acc_list'  ] = []

os.makedirs('logs', exist_ok=True)
filename = ""
for arg in vars(args):
    filename += arg + ':' + str(getattr(args,arg)) + '-'
filename = os.path.join('logs', filename[:-1] + '.pickle')

for epoch in range(args.epochs):
    print("\nEpoch", epoch)
    print("==================================="*2)
    if args.distribute: train_loader.sampler.set_epoch(epoch)

    train_loss, train_acc, log = loop(net, train_loader,  agd,  train=True  )
    test_loss,   test_acc,   _ = loop(net, test_loader,   None, train=False )

    print("Log term:  \t", log        )
    print("Train loss:\t", train_loss )
    print("Test loss: \t", test_loss  )
    print("Train acc: \t", train_acc  )
    print("Test acc:  \t", test_acc   )

    results['log_list'       ].append( log        )
    results['train_loss_list'].append( train_loss )
    results['test_loss_list' ].append( test_loss  )
    results['train_acc_list' ].append( train_acc  )
    results['test_acc_list'  ].append( test_acc   )

    pickle.dump(results, open( filename, "wb" ) )
