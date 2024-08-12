import torch
import gorilla
import argparse
from tqdm import tqdm

from msta3d.model import MSTA3D
from msta3d.evaluation import ScanNetEval
from msta3d.dataset import build_dataloader, build_dataset
from msta3d.utils import get_root_logger, save_gt_instances, save_pred_instances

def get_args():

    parser = argparse.ArgumentParser('MSTA3D')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    args = parser.parse_args()
    return args

def main():

    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    model = MSTA3D(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts, coord_floats = [], [], [], [], []

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            result = model(batch, mode='predict')
            results.append(result)
            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])
        coord_floats.append(res['coord_float'])

    if not cfg.data.test.prefix == 'test':
        logger.info('Evaluate instance segmentation & object detection')
        if cfg.data.test.type == "scannet200":
            scannet_eval = ScanNetEval(dataset.CLASSES, dataset_name=cfg.data.test.type)
        else:
            scannet_eval = ScanNetEval(dataset.CLASSES)
        scannet_eval.evaluate(pred_insts, gt_insts, coord_floats)

    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)

if __name__ == '__main__':
    main()