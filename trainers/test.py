import os
import sys
import torch
import torchvision as tv
from PIL import Image
from dataset_loaders.dataset_utils import IMAGENET_MEAN, IMAGENET_STD,  LabelToLongTensor, EmptyTransform
from dataset_loaders import DAVIS17V2, YTVOSV2
import evaluation
import pandas as pd
import numpy as np
from davis2017.evaluation import DAVISEvaluation
import logging
from time import time

def eval(args, epoch, set, DBtype):
    time_start = time()

    logger = logging.getLogger('global')

    csv_name_global = f'G_results-{set}_{epoch}.csv'
    csv_name_per_sequence = f'Seq_results-{set}_{epoch}.csv'

    csv_name_global_path = os.path.join(args.save_dir, csv_name_global)
    csv_name_per_sequence_path = os.path.join(args.save_dir, csv_name_per_sequence)

    logger.info(f'Evaluating sequences for the semi-supervised task...')
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=args.Davis_dir, task='semi-supervised', gt_set=set, DBtype=DBtype)

    if DBtype=='2016':
        maskDir = os.path.join(args.save_dir,"DAVIS16_val")
    else:
        maskDir = os.path.join(args.save_dir,"DAVIS17_val")

    metrics_res = dataset_eval.evaluate(maskDir)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = ['J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall', 'F-Decay']
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([final_mean, np.mean(J["M"]), np.mean(J["R"]), np.mean(J["D"]), np.mean(F["M"]), np.mean(F["R"]),
                      np.mean(F["D"])])
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    # with open(csv_name_global_path, 'w') as f:
    #     table_g.to_csv(f, index=False, float_format="%.3f")
    # logger.info(f'Global results saved in {csv_name_global_path}')

    # Generate a dataframe for the per sequence results
    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(zip(seq_names, J_per_object, F_per_object)), columns=seq_measures)

    # with open(csv_name_per_sequence_path, 'w') as f:
    #     table_seq.to_csv(f, index=False, float_format="%.3f")
    # logger.info(f'Per-sequence saved in {csv_name_per_sequence_path}')


    # Print the results

    logger.info("-----------------Global results for DAVIS{} ---------------".format(DBtype))
    logger.info(table_g.to_string(index=False))
    if args.TestInfo:
        logger.info("\n---------- Per sequence results for DAVIS{} ----------".format(DBtype))
        logger.info(table_seq.to_string(index=False))
    total_time = time() - time_start
    logger.info('Total time:' + str(total_time))

    return table_g.T[0][0]


def test_model(model, epoch, args):
    logger = logging.getLogger('global')

    nframes = 128
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def image_read(path,size):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [tv.transforms.ToTensor(),
             tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
        return transform(pic)

    def label_read(path,size):
        pic = Image.open(path)
        transform = tv.transforms.Compose(
            [LabelToLongTensor()])
        label = transform(pic)
        return label

    op = EmptyTransform() # Only horizontal flip for first training ( see the paper )
    if args.debug:
        datasets = {
            'DAVIS17_val': DAVIS17V2(args.Davis_dir, '2017', 'val', image_read, label_read, op, nframes,
                                     cacheDir=args.cache_dir)}
    else:
        datasets = {
            'DAVIS17_val': DAVIS17V2(args.Davis_dir, '2017', 'val', image_read, label_read, op, nframes,
                                     cacheDir=args.cache_dir),
            'DAVIS16_val': DAVIS17V2(args.Davis_dir, '2016', 'val', image_read, label_read, op, nframes,
                                     cacheDir=args.cache_dir)
        }

    for key, dataset in datasets.items():

        evaluator = evaluation.VOSEvaluator(dataset, DEVICE, True, debug=args.debug)
        if key == 'DAVIS16_val':
            evaluator = evaluation.VOSEvaluator(dataset, DEVICE, True, debug=args.debug)
            # logger.info("\n-- DAVIS16 dataset initialization started.")
        elif key == 'DAVIS17_val':
            evaluator = evaluation.VOSEvaluator(dataset, DEVICE, True, debug=args.debug)
            # logger.info("\n-- DAVIS17 dataset initialization started.")
        result_fpath = os.path.join(args.save_dir)
        evaluator.evaluate(model, os.path.join(result_fpath, key))

    if args.debug:
        iou2017 = eval(args, epoch, 'val', '2017')
        return iou2017

    else:
        iou2017 = eval(args, epoch, 'val', '2017')
        iou2016 = eval(args, epoch, 'val', '2016')
        return iou2017, iou2016
