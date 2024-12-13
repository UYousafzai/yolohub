import torch
import tqdm
import os
from yolohub.utils.dataset_utils import DefaultDataset
from yolohub.utils.general import compute_ap, non_max_suppression, wh2xy, compute_metric

data_dir = '../Dataset/COCO'

@torch.no_grad()
def test(args, params, model=None):
    filenames = []
    with open(f'{data_dir}/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = os.path.basename(filename.rstrip())
            filenames.append(f'{data_dir}/images/val2017/' + filename)

    dataset = DefaultDataset(filenames, args.input_size, params, augment=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4,
                                         pin_memory=True, collate_fn=DefaultDataset.collate_fn)

    plot = False
    if not model:
        plot = True
        model = torch.load(f='./weights/best.pt', map_location='cuda')
        model = model['model'].float().fuse()

    model.half()
    model.eval()

    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()
    n_iou = iou_v.numel()

    m_pre, m_rec, map50, mean_ap = 0,0,0,0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    for samples, targets in p_bar:
        samples = samples.cuda().half() / 255.
        _, _, h, w = samples.shape
        scale = torch.tensor((w, h, w, h)).cuda()
        outputs = model(samples)
        outputs = non_max_suppression(outputs)
        for i, output in enumerate(outputs):
            idx = targets['idx'] == i
            cls = targets['cls'][idx].cuda()
            box = targets['box'][idx].cuda()

            metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

            if output.shape[0] == 0:
                if cls.shape[0]:
                    metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                continue
            if cls.shape[0]:
                target = torch.cat((cls, wh2xy(box) * scale), dim=1)
                metric = compute_metric(output[:, :6], target, iou_v)
            metrics.append((metric, output[:, 4], output[:, 5], cls.squeeze(-1)))

    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = compute_ap(*metrics, plot=plot, names=params["names"])

    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    model.float()
    return mean_ap, map50, m_rec, m_pre
