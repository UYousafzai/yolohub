import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data

from yolohub.models.yolov11 import yolo_v11_n
from yolohub.utils.general import (setup_seed, setup_multi_processes, set_params, EMA, strip_optimizer, LinearLR, 
                                   AverageMeter, ComputeLoss)
from yolohub.utils.dataset_utils import DefaultDataset
from yolohub.test import test

warnings.filterwarnings("ignore")

data_dir = '../Dataset/COCO'

def train(args, params):
    model = yolo_v11_n(len(params['names']))
    model.cuda()

    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    optimizer = torch.optim.SGD(set_params(model, params['weight_decay']),
                                params['min_lr'], params['momentum'], nesterov=True)

    ema = EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open(f'{data_dir}/train2017.txt') as reader:
        for filename in reader.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/train2017/' + filename)

    sampler = None
    dataset = DefaultDataset(filenames, args.input_size, params, augment=True)

    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    loader = data.DataLoader(dataset, args.batch_size, shuffle=(sampler is None), sampler=sampler,
                             num_workers=8, pin_memory=True, collate_fn=DefaultDataset.collate_fn)

    num_steps = len(loader)
    scheduler = LinearLR(args, params, num_steps)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    best = 0
    amp_scale = torch.amp.GradScaler()
    criterion = ComputeLoss(model, params)

    if args.local_rank == 0 and not os.path.exists('weights'):
        os.makedirs('weights')

    with open('weights/step.csv', 'w', newline='') as log:
        if args.local_rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()

        for epoch in range(args.epochs):
            model.train()
            if args.distributed and sampler is not None:
                sampler.set_epoch(epoch)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            p_bar = enumerate(loader)

            if args.local_rank == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad(set_to_none=True)
            avg_box_loss = AverageMeter()
            avg_cls_loss = AverageMeter()
            avg_dfl_loss = AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() / 255

                with torch.amp.autocast('cuda'):
                    outputs = model(samples)
                    loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= (args.batch_size * args.world_size)
                loss_cls *= (args.batch_size * args.world_size)
                loss_dfl *= (args.batch_size * args.world_size)

                amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                if step % accumulate == 0:
                    amp_scale.step(optimizer)
                    amp_scale.update()
                    optimizer.zero_grad(set_to_none=True)
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                if args.local_rank == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args.epochs}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args.local_rank == 0:
                last = test(args, params, ema.ema)

                logger.writerow({
                    'epoch': str(epoch + 1).zfill(3),
                    'box': f'{avg_box_loss.avg:.3f}',
                    'cls': f'{avg_cls_loss.avg:.3f}',
                    'dfl': f'{avg_dfl_loss.avg:.3f}',
                    'mAP': f'{last[0]:.3f}',
                    'mAP@50': f'{last[1]:.3f}',
                    'Recall': f'{last[2]:.3f}',
                    'Precision': f'{last[3]:.3f}'
                })
                log.flush()

                if last[0] > best:
                    best = last[0]

                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                torch.save(save, f='./weights/last.pt')
                if best == last[0]:
                    torch.save(save, f='./weights/best.pt')
                del save

    if args.local_rank == 0:
        strip_optimizer('./weights/best.pt')
        strip_optimizer('./weights/last.pt')

    torch.cuda.empty_cache()


def profile(args, params):
    import thop
    from yolohub.models.yolov11 import yolo_v11_n

    shape = (1, 3, args.input_size, args.input_size)
    model = yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape).cuda())

    x = torch.empty(shape).cuda()
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args.local_rank == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')


def main():
    parser = ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    if args.distributed:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open('yolohub/configs/yolov11.yaml', 'r') as f:
        params = yaml.safe_load(f)

    setup_seed()
    setup_multi_processes()

    profile(args, params)

    if args.train:
        train(args, params)
    if args.test:
        test(args, params)


if __name__ == "__main__":
    main()
