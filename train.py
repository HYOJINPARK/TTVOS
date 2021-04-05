import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import cv2
import torch


import models
import argparse

from utils.Tensor_logger import Logger
from torch.utils.collect_env import get_pretty_env_info
import datetime
from utils.log_helper import init_log, add_file_handler
import logging
from utils.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from shutil import copyfile
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from trainers.train_YTB import train_YTB
from trainers.train_DAVIS import train_DAVIS
from trainers.train_Saliency import train_Saliency


parser = argparse.ArgumentParser(description='PyTorch Tracking VOS Training')
# /media/hyojin/SSD1TB/Dataset
parser.add_argument('--Saliency_batch', type=int, help='Batch size', default=48)
parser.add_argument('--epochs_Saliency', type=int, help='Number of epochs for first training', default=100)
parser.add_argument("--Saliency",type=bool, default=False,help=" Using Saliency dataset for train ", choices=[True, False])
parser.add_argument("--use_tps",type=bool, default=True,help=" Using TPS transform for train ")
parser.add_argument("--color_aug",type=bool, default=False,help=" Using color_aug transform for train ")
parser.add_argument("--nb_points",type=int, default=5,help=" Using color_aug transform for train ")
parser.add_argument('--STrain_fN', default=[3], help='Num of train frame for Saliency dataset ')
parser.add_argument('--SaliencySize', default=(240, 432), help='Training Saliency Training Size')
parser.add_argument('--lr_Saliency', default=2e-4, type=float, help='learning rate(default: 1e-4)')
parser.add_argument('--lrSch_Saliency', default='Not', choices=['Exp', 'WarmP', 'Step', 'Not'])

parser.add_argument('--YTB_batch', type=int, help='Batch size', default=24)
parser.add_argument('--epochs_Ytb', type=int, help='Number of epochs for first training', default=100)
parser.add_argument("--YTB",type=bool, default=False,help=" Using YTB dataset for train ", choices=[True, False])
parser.add_argument("--YTB_trans",type=bool, default=True,help=" Using Affine transform for train ")
parser.add_argument('--YTrain_fN', default=[8], help='Num of train frame for youtube dataset ')
parser.add_argument('--YtbSize', default=(240, 432), help='Training Youtube Training Size')
parser.add_argument('--lr_YTB', default=3e-4, type=float, help='learning rate(default: 1e-4)')
parser.add_argument('--lrSch_YTB', default='Not', choices=['Exp', 'WarmP', 'Mstep', 'Not'])


parser.add_argument('--DAVIS_batch', type=int, help='Batch size', default=8)
parser.add_argument('--epochs_Davis', type=int, help='Number of epochs for second trainig', default=100)
parser.add_argument("--DAVIS",type=str, default=True,help=" Using DAVIS dataset for train ",choices=[True, False])
parser.add_argument("--DAVISversion",type=str, default='All',help="DAVIS version for train ",choices=['2016', '2017', 'All'])
parser.add_argument("--DAVIS_trans",type=bool, default=True,help=" Using Affine transform for train ")
parser.add_argument('--DTrain_fN', default=[8], help='Num of train frame for Davis dataset')
parser.add_argument('--DavisSize', default=(480, 864), help='Training Davis Training Size')
parser.add_argument('--lr_DAVIS', default=1e-4, type=float, help='learning rate(default: 1e-4)')
parser.add_argument('--lrSch_DAVIS', default='Not', choices=['Exp', 'WarmP', 'Mstep', 'Not'])

# Loading setting
parser.add_argument('--save_dir', default='./save_dir', type=str, help='save dir')
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache dir')
parser.add_argument('--saliency_dir', default='/media/hyojin/SSD1TB1/Dataset/Saliency', type=str, help='Ytb dataset dir')
parser.add_argument('--Ytb_dir', default='/media/hyojin/SSD1TB1/Dataset/Youtube-VOS2019', type=str, help='Ytb dataset dir')
parser.add_argument('--Davis_dir', default='/media/hyojin/SSD1TB1/Dataset//DAVIS', type=str, help='Davis dataset dir')
parser.add_argument('--nnWeight', default='./nnWeight', type=str, help='nn_weights_path')

#Model Strtucture
parser.add_argument('--backbone', default='hrnetv2Sv1', choices=['resnet50s16', 'hrnetv2Sv1'],
                    help='architecture of backbone model')
parser.add_argument('--refine', default='v3', choices=['v1', 'v2', 'v3'])

parser.add_argument('--LT', default='slim', choices=['Not', 'gc', 'slim'])
parser.add_argument('--ltG', default=4, type=int, help="number of group in long_term module")
parser.add_argument('--ltLoc', default='s8', type=str, help="input feature for long-term")

#Train setting
parser.add_argument('--parallel', default=False, type=bool, help='use parallel')
parser.add_argument('--Opt', default='Adam', choices=['Adam', 'SGD'])
parser.add_argument("--loss_type",type=str, default="NLL", choices=["NLL", "ohem", "ohemEdge"])

parser.add_argument('--DiffLoss', default=True, choices=[True, False])
parser.add_argument('--TmpLoss', default=True, choices=[True, False])
parser.add_argument("--loss_W", default=[1.0, 1.0,1.0],help='Loss vos Diff, Tmp loss')
parser.add_argument('--wd', default=0, type=float, help='weight decay (default: 1e-5)')
parser.add_argument('--MultiLR', default=False,choices=[True, False])
parser.add_argument('--Multi_W', default=[1.0, 10.0])

parser.add_argument('--freeze', default=True,  help='free in resNet')
parser.add_argument('--train_layer', default=('stage4',),  help='free in backbone like layer4 , stage4 ')

# Etc/
parser.add_argument("--resume",type=str, default="",help="Resume from First Train")
parser.add_argument("--debug",type=bool, default=False)
parser.add_argument("--log",type=str, default="log.txt",help="save printed log ")
parser.add_argument("--TestInfo", default=False,help="show detail eval result")
parser.add_argument("--pth", default="",help="get pth file")
parser.add_argument('--print_interval', default=500, type=int, help='print info interval')
parser.add_argument("--val_type", default="trainOnly",help="both, trainOnly, officialOnly, valOnly ")

def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str

def main():

    args = parser.parse_args()

    now = datetime.datetime.now()
    time_str = now.strftime("%m-%d_%H%M%S")
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    args.save_dir = args.save_dir + "/" + time_str
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)

    init_log('global', logging.INFO)
    args.log = os.path.join(args.save_dir, args.log)
    copy_name = ["train.py", "VOS_model.py", "TTVOS_model.py", "Dnc_models.py", "vos_trainer.py",
                 "ytvos_v2.py", "davis17_v2.py"]
    copyfile(copy_name[0], os.path.join(args.save_dir, copy_name[0]))
    copyfile(os.path.join("models", copy_name[1]), os.path.join(args.save_dir, copy_name[1]))
    copyfile(os.path.join("models", copy_name[2]), os.path.join(args.save_dir, copy_name[2]))
    copyfile(os.path.join("models", copy_name[3]), os.path.join(args.save_dir, copy_name[3]))
    copyfile(os.path.join("trainers", copy_name[4]), os.path.join(args.save_dir, copy_name[4]))


        ######## str to bool ##########################

    tb_logger = Logger(8097, args.save_dir)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)
    logger = logging.getLogger('global')

    if args.backbone == "resnet50s16" and (args.freeze==False) :

        if args.freeze:
            _model = models.VOS( backbone=('resnet50s16', (True, args.train_layer, ('layer4',), ('layer2',), ('layer1',),
                                          args.nnWeight)), mode='train', args=args)
        else:
            _model = models.VOS(backbone=('resnet50s16', (True, ('layer4', 'layer3', 'layer2', 'layer1'),
                              ('layer4',), ('layer2',), ('layer1',),args.nnWeight)), mode='train', args=args)
            logger.info("Train every layers of ResNet")


    elif "hrnet" in args.backbone :

        if args.freeze:
            _model = models.VOS(backbone=(args.backbone, (True, args.train_layer,args.nnWeight)), mode='train', args=args)
        else:
            _model = models.VOS(backbone=(args.backbone, (True, ('Not',), args.nnWeight)), mode='train', args=args)
            logger.info("Train every layers of HRNet")



    print(_model)

    logger.info("\n" + collect_env_info())
    logger.info(args)
    logger.info("Start training ...")
    logger.info("Using : ")
    logger.info("    Loss_W : " + str(args.loss_W))
    logger.info("    MultiLR : " + str(args.MultiLR))
    logger.info("    Optimizer : " + (args.Opt))
    logger.info("    WeightDecay : " + str(args.wd))
    logger.info("    Backbone : " + (args.backbone))
    logger.info("    Refine : " + (args.refine))
    logger.info("    LT : " + (args.LT))
    logger.info("    DiffLoss : " + str(args.DiffLoss))
    logger.info("    TmpLoss : " + str(args.TmpLoss))




    if args.Saliency:
        logger.info("\n")
        logger.info("    Saliency lrSch : " + (args.lrSch_Saliency))
        logger.info("    Saliency LearningRate : " + str(args.lr_Saliency))
        logger.info("    Epochs Saliency training : " + str(args.epochs_Saliency))
        logger.info("    Batch size Saliency : " + str(args.Saliency_batch))
        logger.info("    Saliency Train frameN : " + str(args.STrain_fN))
        logger.info("    TSP Transform : " + str(args.use_tps))
        logger.info("    ColorAug Transform : " + str(args.color_aug))

    if args.YTB:
        logger.info("\n")
        logger.info("   YTB lrSch : " + (args.lrSch_YTB))
        logger.info("    YTB LearningRate : " + str(args.lr_YTB))
        logger.info("    Epochs YTBVos training : " + str(args.epochs_Ytb))
        logger.info("    Batch size YTBVos : " + str(args.YTB_batch))
        logger.info("    Ytb Train frameN : " + str(args.YTrain_fN))
        logger.info("    Ytb affine Transform : " + str(args.YTB_trans))
        copyfile(os.path.join("dataset_loaders", copy_name[5]), os.path.join(args.save_dir, copy_name[5]))

    if args.DAVIS:
        logger.info("\n")
        logger.info("    DAVIS lrSch : " + (args.lrSch_DAVIS))
        logger.info("    DAVIS LearningRate : " + str(args.lr_DAVIS))
        logger.info("    Davis Version : " + args.DAVISversion)
        logger.info("    Epochs Davis training : " + str(args.epochs_Davis))
        logger.info("    Batch size Davis : " + str(args.DAVIS_batch))
        logger.info("    Davis Train frameN : " + str(args.DTrain_fN))
        logger.info("    Davis affine Transform : " + str(args.DAVIS_trans))
        copyfile(os.path.join("dataset_loaders", copy_name[6]), os.path.join(args.save_dir, copy_name[6]))


    x = torch.rand(1,2, 3,480, 854)
    gt = torch.zeros(1,1,480, 854)
    gt[:,:,100:200,100:200] = 1
    gt=gt.long()
    gt_set=[gt,None]
    # gt_set=[gt]

    _model.train(False)
    model_eval = add_flops_counting_methods(_model)
    model_eval.start_flops_count()
    _ = model_eval(x, gt_set, None)

    N_flop = _model.compute_average_flops_cost()
    logger.info("input size is : {}".format(str(x.size())))
    logger.info('Flops:  {}'.format(flops_to_string(N_flop)))
    logger.info('Params: ' + get_model_parameters_number(_model))


    _model.to(DEVICE)
    if args.pth == "":
        pthname = None
    else:
        pthname = args.pth

    if args.Saliency:
        logger.info("------------ start Saliency VOS training --------------")

        model = _model
        if pthname is not None:
            model.load_state_dict(torch.load(pthname, map_location=torch.device(DEVICE))['net'])
            logger.info("Take pth file : {}".format(pthname))

        if torch.cuda.device_count()>1 and args.parallel:
            model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
        else:
            model = model.cuda()

        opt_ep, prefix = train_Saliency(model, args, tb_logger)
        pthname = '{}_ep{:04d}.pth.tar'.format(prefix, args.epochs_Saliency)
        logger.info("-----end Saliency VOS training Best: {}-----".format(pthname))

    args.val_type = 'officialOnly'
    if args.YTB:
        logger.info("------------ start YTB VOS training --------------")

        model = _model
        if pthname is not None:
            model.load_state_dict(torch.load(pthname, map_location=torch.device(DEVICE))['net'])
            logger.info("Take pth file : {}".format(pthname))

        if torch.cuda.device_count()>1 and args.parallel:
            model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
        else:
            model = model.cuda()

        opt_ep, prefix = train_YTB(model, args, tb_logger)
        pthname = '{}_ep{:04d}.pth.tar'.format(prefix, opt_ep)
        logger.info("-----end YTB VOS training Best: {}-----".format(pthname))


    if args.DAVIS:
        logger.info("----------- start DAVIS VOS training -------------")
        model = _model
        if pthname is not None:
            model.load_state_dict(torch.load(pthname, map_location=torch.device(DEVICE))['net'])
            logger.info("Take pth file : {}".format(pthname))


        if torch.cuda.device_count() > 1 and args.parallel:
            model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
        else:
            model = model.cuda()
        opt_ep, prefix = train_DAVIS(model,args, tb_logger)
        pthname = '{}_ep{:04d}.pth.tar'.format(prefix, opt_ep)
        logger.info("---end DAVIS training Best: {}------".format(pthname))

if __name__ == '__main__':
    main()
