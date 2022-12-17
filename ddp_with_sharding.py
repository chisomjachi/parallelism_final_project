import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
from ResNet import Bottleneck, ResNet, ResNet50
from time import time
import numpy as np
import matplotlib.pyplot as plt
from visualizer import *
from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
from fairscale.optim.oss import OSS

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def train(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank)
    torch.manual_seed(0)
    model = ResNet50(10)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 120
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    base_optimizer_arguments = { "lr": 1e-4,"momentum":0.9}
    base_optimizer = torch.optim.SGD
    optimizer = OSS(params=model.parameters(),optim=base_optimizer,**base_optimizer_arguments)
    optimizer.consolidate_state_dict()
    # Wrap the model
    
    # model = ShardedDDP(model, optimizer)
    
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    
    model = DDP(model)
    # Data loading code
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = torchvision.datasets.CIFAR10(root='data/',
                                             train=True,
                                             download=True,
                                             transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=train_sampler
    )

    start = datetime.now()
    total_step = len(train_loader)
    train_losses = []
    training_times = []
    for epoch in range(args.epochs):
        t1 = time()
        epoch_train_losses = []
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_value = loss.item()
            epoch_train_losses.append(loss_value)
            # Backward and optimize
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                )
        t2 = time()
        training_times.append(t2-t1)
        train_losses.append(np.mean(epoch_train_losses))
    if gpu == 0:
        means = np.mean(training_times)
        stds = np.std(training_times)
        plot([means],[stds], ['DDP with 4 GPUs'],'ddp_withshariding_mean_time.png') 
        plot_training_no_val(train_losses,training_times, setup = 'ddp_with_sharding')
        # print(train_losses)
        # plt.savefig('books_read.png')
        print("Training complete in: " + str(datetime.now() - start))
        print("Training Time Mean: " + str(np.mean(training_times)))
        # print(training_times)


if __name__ == '__main__':
    main()