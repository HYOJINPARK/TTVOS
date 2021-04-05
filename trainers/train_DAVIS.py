import os
import torch
from dataset_loaders import DAVIS17V2
from dataset_loaders.dataset_utils import ConcatDataset, random_object_sampler, deterministic_object_sampler, MyBatchSampler
from utils.Loss_functions import OhemNLLLoss, OhemNLLEdgeLoss, NLLLoss2d
from utils.lr_helper import WarmupPoly

from torch.utils.data import DataLoader
import dataset_loaders
import multiprocessing
import trainers

import logging
logger = logging.getLogger('global')


def train_DAVIS(model, args, tb_logger=None):
    logger = logging.getLogger('global')
    batch_size = args.DAVIS_batch
    num_epochs = args.epochs_Davis

    if args.DAVIS_trans:
        logger.info("Applying affine transformation to DAVIS")
        op = dataset_loaders.dataset_utils.JointImageVidTransform((-30, 30), scale=(0.75, 1.25), shear=(-30, 30))
        # op = dataset_loaders.JointImageVidTransform((-15, 15), scale=(0.75, 1.25), shear=(-15, 15))
    else:
        op = dataset_loaders.dataset_utils.EmptyTransform()  # Only horizontal flip for first training ( see the paper )
    # op = None
    op_image_init = dataset_loaders.dataset_utils.train_image_init_read
    op_label_init = dataset_loaders.dataset_utils.train_label_init_read
    val_op = None
    nb_workers = multiprocessing.cpu_count()//2

    if len(args.DTrain_fN) == 1:
        if args.DAVISversion == 'All':
            train_set_1 = DAVIS17V2(args.Davis_dir, "2016", 'train', op_image_init, op_label_init, op,
                                    args.DTrain_fN[0], random_object_sampler, start_frame="random", size=args.DavisSize,
                                    cacheDir=args.cache_dir)
            train_set_2 = DAVIS17V2(args.Davis_dir, "2017", 'train', op_image_init, op_label_init, op,
                                    args.DTrain_fN[0], random_object_sampler, start_frame="random", size=args.DavisSize,
                                    cacheDir=args.cache_dir)

            train_set = ConcatDataset(train_set_1, train_set_2, batch_size)
            train_set = ConcatDataset(train_set_1, train_set, batch_size)

            train_loader = DataLoader(train_set,
                                      batch_sampler=MyBatchSampler(batch_size, len(train_set_1), len(train_set_2)),
                                      num_workers=nb_workers)
        else:
            train_set = DAVIS17V2(args.Davis_dir, args.DAVISversion, 'train', op_image_init, op_label_init, op,
                                  args.DTrain_fN[0], random_object_sampler, start_frame="random", size=args.DavisSize,
                                  cacheDir=args.cache_dir)
            train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=nb_workers)


    else:
        train_set_1 = DAVIS17V2(args.Davis_dir, args.DAVISversion, 'train', op_image_init, op_label_init, op,
                                args.DTrain_fN[0], random_object_sampler, start_frame="random", size=args.DavisSize,
                                cacheDir=args.cache_dir)
        train_set_2 = DAVIS17V2(args.Davis_dir, args.DAVISversion, 'train', op_image_init, op_label_init, op,
                     args.DTrain_fN[1], random_object_sampler, start_frame="random",size=args.DavisSize,
                                cacheDir=args.cache_dir)
        train_set = ConcatDataset(train_set_1, train_set_2, batch_size)
        train_loader = DataLoader(train_set, batch_sampler=MyBatchSampler(batch_size, len(train_set_1), len(train_set_2)),
                                  num_workers=nb_workers)
    if args.val_type !='officialOnly':
        val_set = DAVIS17V2(args.Davis_dir, '2016', 'val', op_image_init, op_label_init, val_op, args.DTrain_fN[0],
                              deterministic_object_sampler, start_frame='first', size=args.DavisSize,
                             cacheDir=args.cache_dir)

        val_loader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=nb_workers)
        logger.info("Sets initiated with {} (train) and {} (val) samples.".format(len(train_set), len(val_set)))
    else:
        val_loader = None
        logger.info("Sets initiated with {} (train) samples.".format(len(train_set)))


    if args.loss_type == "NLL":
        loss_w = torch.FloatTensor([0.2,0.8])
        objective = NLLLoss2d(ignore_index=255)
    elif args.loss_type == "ohem":
        objective = OhemNLLEdgeLoss(ignore_index=255, edge_loss_scale=args.els)
    elif args.loss_type == "ohemEdge":
        objective = OhemNLLEdgeLoss(ignore_index=255, min_kept_ratio=1., edge_loss_scale=args.els)
    else:
        logger.info("Wrong selection change to default loss NLL")
        objective = NLLLoss2d(ignore_index=255)
    if torch.cuda.is_available():
        objective = objective.cuda()

    if args.Opt == "Adam":
        optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=args.lr_DAVIS,
                                     weight_decay=args.wd)
    elif args.Opt == "SGD":
        optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad], lr=args.lr_DAVIS,
                                    weight_decay=args.wd)
    if args.lrSch_DAVIS == "Not":
        lr_sched = None
        step_type = "batch"
    elif "Exp" in args.lrSch_DAVIS:
        lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.995)
        step_type = "batch"
        print("decay by exponentially")
    elif "WarmP" in args.lrSch_DAVIS:
        lr_sched = WarmupPoly(init_lr=args.lr_DAVIS, total_iter=(args.epochs_Davis+1)*len(train_loader))
        step_type = "iter"
        if args.Opt == "Adam":
            optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad],
                                         lr=lr_sched.get_lr(1), weight_decay=args.wd)
        elif args.Opt == "SGD":
            optimizer = torch.optim.SGD([param for param in model.parameters() if param.requires_grad],
                                        lr=lr_sched.get_lr(1), weight_decay=args.wd)
    elif "Mstep" in args.lrSch_DAVIS :
        decay1 = args.epochs_Davis // 2
        decay2 = args.epochs_Davis - args.epochs_Davis // 6
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[decay1, decay2], gamma=0.5)
        step_type = "batch"
        print("decay by multiStep")

    trainer = trainers.VOSTrainer(
        model, optimizer, objective, lr_sched, step_type, train_loader, val_loader, workspace_dir=args.save_dir,
        save_name=os.path.splitext(os.path.basename(__file__))[0]+"_davis",  print_interval=args.print_interval,
        debug=args.debug,args=args, tb_logger=tb_logger)
    if args.resume !="":
        trainer.load_checkpoint(args.resume)
    opt_ep, prefix = trainer.train(num_epochs)
    return opt_ep, prefix

