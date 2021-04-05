import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch

import argparse
from trainers.test import test_model

from utils.log_helper import init_log, add_file_handler
import logging
from utils.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
import models

import importlib
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='PyTorch VOS Training')
# /media/hyojin/SSD1TB/Dataset
#data_loading
parser.add_argument('--cache_dir', default='./cache', type=str, help='cache dir')
parser.add_argument('--Ytb_dir', default='/media/hyojin/SSD1TB1/Dataset/Youtube-VOS2019', type=str, help='Ytb dataset dir')
parser.add_argument('--Davis_dir', default='/media/hyojin/SSD1TB1/Dataset//DAVIS', type=str, help='Davis dataset dir')

parser.add_argument('--nnWeight', default='./nnWeight', type=str, help='nn_weights_path')

#Model Strtucture
parser.add_argument('--backbone', default='hrnetv2Sv1', choices=['resnet50s16', 'hrnetv2Sv1'], help='architecture of backbone model')
parser.add_argument('--refine', default='v3', choices=['v1', 'v2', 'v3'])
parser.add_argument('--DiffLoss', default=True, choices=[True, False])
parser.add_argument('--TmpLoss', default=True, choices=[True, False])

parser.add_argument('--LT', default='slim', choices=['Not', 'gc', 'slim'])
parser.add_argument('--ltG', default=4, type=int, help="number of group in slim")
parser.add_argument('--ltLoc', default='s8', type=str, help="input feature for for long-term")

# Etc
parser.add_argument('--parallel', default=False, type=bool, help='use parallel')
parser.add_argument("--debug",type=bool, default=False,help="Vis out detail feature map")
parser.add_argument("--log",type=str, default="logTest.txt",help="Out log file")
parser.add_argument("--TestInfo", default="False",help="show detail eval result")
parser.add_argument('--save_dir', default='./save_dir/', type=str, help='save dir')
parser.add_argument("--pth", default="TTVOS.pth.tar",help="get pth file")

def main():

    args = parser.parse_args()
    init_log('global', logging.INFO)

        ######## str to bool ##########################


    args.TestInfo = (args.TestInfo =="True")
    logger = logging.getLogger('global')


    if args.backbone == "resnet50s16":
        _model = models.VOS(backbone=('resnet50s16', (True, ('layer4',), ('layer4',), ('layer2',), ('layer1',),
                                        args.nnWeight)), mode='eval', args=args)

    elif "hrnet" in args.backbone :

        _model = models.VOS(backbone=(args.backbone, (True, ('stage4',),args.nnWeight)), mode='eval', args=args)

    # print(_model)

    logger.info(args)
    logger.info("Start training ...")
    logger.info("Using : ")

    logger.info("    Backbone : " + (args.backbone))
    logger.info("    Refine : " + (args.refine))
    logger.info("    LT : " + (args.LT))


    x = torch.rand(1, 2, 3, 480, 854)
    gt = torch.zeros(1, 1, 480, 854)
    gt[:, :, 100:200, 100:200] = 1
    gt = gt.long()
    gt_set = [gt, None]

    _model.train(False)
    model_eval = add_flops_counting_methods(_model)
    model_eval.start_flops_count()
    model_eval(x, gt_set, None)

    N_flop = _model.compute_average_flops_cost()
    logger.info("input size is : {}".format(str(x.size())))
    logger.info('Flops:  {}'.format(flops_to_string(N_flop)))
    logger.info('Params: ' + get_model_parameters_number(_model))

    _model.to(DEVICE)
    if args.pth == "":
        pthname = None
    else:
        pthname = os.path.join(args.save_dir,args.pth)
    args.save_dir = os.path.join(args.save_dir,"DEBUG")
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    args.log = os.path.join(args.save_dir, args.log)

    if args.log != "":
        add_file_handler('global', args.log, logging.INFO)

    model = _model
    if pthname is not None:
        model.load_state_dict(torch.load(pthname, map_location=torch.device(DEVICE))['net'])
        logger.info("Take pth file : {}".format(pthname))

    if torch.cuda.device_count() > 1 and args.parallel:
        model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
    else:
        model = model.cuda()

    test_model(model, 0, args)


if __name__ == '__main__':
    main()