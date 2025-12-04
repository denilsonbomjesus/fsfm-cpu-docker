import argparse
import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Adjust import paths to be relative to the project root
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Corrected imports
from fas import fas_model_fix
from util.datasets import build_dataset
import util.misc as misc
from utils.utils import accuracy as accuracy_fas

def get_args_parser():
    parser = argparse.ArgumentParser('FAS Finetuning', add_help=False)
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Path to save logs and checkpoints')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--model', type=str, default='vit_small_patch16', help='Name of the model to train')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    # Add other necessary arguments from fas_model_fix if needed
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')
    parser.add_argument('--pt_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--drop_path', type=float, default=0.0, help='Drop path rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # Dummy arguments to be compatible with build_dataset
    parser.add_argument('--normalize_from_IMN', action='store_true')
    parser.add_argument('--apply_simple_augment', action='store_true')
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    return parser

@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    for images, labels in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type):
            cls_out, _ = model(images)
            loss = criterion(cls_out.float(), labels)

        acc1, = accuracy_fas(cls_out, labels, topk=(1,))
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch(model, data_loader, optimizer, device, criterion, epoch, log_writer=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    for i, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type):
            cls_out, _ = model(images)
            loss = criterion(cls_out.float(), labels)

        loss.backward()
        optimizer.step()

        if device.type == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    device = torch.device(args.device)

    # --- Dataset and DataLoader ---
    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    data_loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # --- Model ---
    model = fas_model_fix(args).to(device)

    # --- Optimizer and Loss ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    
    log_dir = args.output_dir
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=log_dir) if misc.is_main_process() else None


    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, data_loader_train, optimizer, device, criterion, epoch, log_writer)
        val_stats = evaluate(model, data_loader_val, device, criterion)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            misc.save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, checkpoint_path)


    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H:%M:%S', time.gmtime(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FAS Finetuning', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
