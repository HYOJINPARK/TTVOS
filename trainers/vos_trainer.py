import subprocess
from itertools import zip_longest
import glob

import torch
import torch.nn.functional as F
import torchvision as tv
import os
import trainers
import utils
from trainers.test import test_model
import logging
import subprocess
logger = logging.getLogger('global')


def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


class Downsampling_avg(torch.nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = torch.nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(torch.nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''

        for pool in self.pool:
            input = pool(input)

        return input


def get_thislr(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])

    return lr

class VOSTrainer(object):
    def __init__(self, model, optimizer, objective_vos, lr_sched,lr_step, train_loader_vos, val_loader_vos,
                 workspace_dir=None, save_name=None, print_interval=25, debug=False,args=None,
                 tb_logger=None):
        self._model = model
        self._optimizer = optimizer
        self._objective_vos = objective_vos
        self._lr_sched = lr_sched
        self._lr_step = lr_step

        self._train_loader_vos = train_loader_vos
        self._val_loader_vos = val_loader_vos

        if args.val_type == 'officialOnly' or args.val_type == 'trainOnly' :
            self._val_loader_vos=None
        self._parallel = args.parallel

        self.prefix=""
        self.loss_W= args.loss_W
        # Initialize statistics variables
        self._stats = {}
        if self._train_loader_vos is not None:
            self._stats['train_vos loss'] = utils.AverageMeter()
            self._stats['train_vos mIoU'] = utils.AverageMeter()
            self._stats['train_vos Diff'] = utils.AverageMeter()
            self._stats['train_vos Tmp'] = utils.AverageMeter()


        if self._val_loader_vos is not None:
            self._stats['val_vos loss'] = utils.AverageMeter()
            self._stats['val_vos mIoU'] = utils.AverageMeter()
            self._stats['val_vos Diff'] = utils.AverageMeter()
            self._stats['val_vos Tmp'] = utils.AverageMeter()

        self._use_gpu = torch.cuda.is_available()
        self._workspace_dir = workspace_dir
        self._save_name = save_name
        self._debug=debug
        self._print_interval = print_interval
        self.best2017, self.ep2017 =0, 0
        self.best2016, self.ep2016 = 0, 0
        self.bestVal, self.epVal = 0, 0
        self.bestTvos, self.epT =0,0
        self.savePth =False

        self._tb_logger = tb_logger
        self._args=args
        self.DiffLoss = args.DiffLoss
        self.TmpLoss = args.TmpLoss

        self._epoch = 1
        if self._use_gpu:
            self._device = 'cuda'
        else:
            self._device = 'cpu'
        # self._model.to(self._device)
        if args.DiffLoss :
            self.heatmap_loss = torch.nn.MSELoss().to(self._device)
        if args.TmpLoss:
            self.Tmp_loss = torch.nn.BCELoss().to(self._device)




    def train(self, max_epochs):
        self.max_ep=max_epochs+1
        for epoch in range(self._epoch, max_epochs+1):
            self._epoch = epoch
            self.train_epoch()
            if self.savePth or (self._epoch%10==0):
                self.save_checkpoint()
            if self._lr_sched is not None and self._lr_step == "batch":
                lr = get_thislr(self._optimizer)
                for i in range(len(lr)):
                    self._tb_logger.scalar_summary('lr'+str(i+1), lr[i], self._epoch)
                self._lr_sched.step()

            logger.info('BEST info training! until EP {}'.format(epoch))
            logger.info("Best Train mIoU score {} in EP{}".format(self.bestTvos,self.epT))
            logger.info("Best 2017 J&F score {} in EP{}".format(self.best2017,self.ep2017))
            logger.info("Best 2016 J&F score {} in EP{}".format(self.best2016,self.ep2016))
            logger.info("Best val mIoU score {} in EP{}".format(self.bestVal,self.epVal))
            # logger.info(get_gpu_memory_map())
            logger.info("\n\n")

        return max(self.ep2017, self.ep2016, self.epVal), self.prefix


    def train_epoch(self):
        """Do one epoch of training and validation."""
        # JF2017, JF2016 = test_model(self._model, self._epoch, self._args)
        self.savePth = False
        self._model.train(True)
        lr = get_thislr(self._optimizer)
        for i in range(len(lr)):
            logger.info("EP{} : lr{} is {}".format(self._epoch,i+1, lr[i]))

        self.cycle_dataset(mode='train')
        self._tb_logger.scalar_summary('train_loss', self._stats['train_vos loss'].avg, self._epoch)
        self._tb_logger.scalar_summary('train_iou', self._stats['train_vos mIoU'].avg, self._epoch)
        if self.bestTvos < self._stats['train_vos mIoU'].avg:
            self.bestTvos = self._stats['train_vos mIoU'].avg
            self.epT = self._epoch

            # Validation
        if self._val_loader_vos is not None:
            self._model.train(False)
            with torch.no_grad():
                self.cycle_dataset(mode='val')

            if self.bestVal < self._stats['val_vos mIoU'].avg:
                self.epVal = self._epoch
                self.bestVal = self._stats['val_vos mIoU'].avg
                self.savePth = True
            self._tb_logger.scalar_summary('val_loss', self._stats['val_vos loss'].avg, self._epoch)
            self._tb_logger.scalar_summary('val_iou', self._stats['val_vos mIoU'].avg, self._epoch)

        if self._args.val_type != "valOnly" and self._args.val_type != "trainOnly" and self._epoch>(self.max_ep*0.7):
            JF2017, JF2016 = test_model(self._model,self._epoch,self._args)
            self._tb_logger.scalar_summary('JF2017', JF2017, self._epoch)
            self._tb_logger.scalar_summary('JF2016', JF2016, self._epoch)

            if self.best2017 < JF2017:
                self.best2017 = JF2017
                self.ep2017 = self._epoch
                self.savePth = True
            if self.best2016 < JF2016:
                self.best2016 = JF2016
                self.ep2016 = self._epoch
                self.savePth = True


        # Update all stat values
        for stat_value in self._stats.values():
            if isinstance(stat_value, utils.AverageMeter):
                stat_value.new_epoch()

    def cycle_dataset(self, mode):
        """Do a cycle of training or validation.
        Assumptions:
            loader outputs ((images, firstframe_segmentation, None), (perframe_labels, single_label))
            images: (batchsize,samplelen,nchannels,height,width) Tensor corresp to video
            firstframe_label: (batchsize,nclasses,height,width) Tensor with labels for the first frame
            None: gives space for internal states from the network
            perframe_labels: (batchsize,samplelen,height,width) Tensor with labels for all frames
            single_label: (batchsize,height,width) Tensor with pixel values as class
            model output: (batchsize,nclasses,height,width) Tensor corresp to frame segmentation"""
        if mode == 'train':
            loader_vos = self._train_loader_vos
        elif mode == 'val':
            loader_vos = self._val_loader_vos

        if loader_vos is None:
            loader_vos = []

        vos_miou_extremes = [(1.0,-1), (.0,-1)]

        for i, vos_data in enumerate(loader_vos):
            # Read data

            #print ("Images shape : " + str(vos_data['images'].shape))
            #print ("given segannos shape : " + str(vos_data['given_seganno'].shape))
            #print ("segannos shape : " + str(vos_data['segannos'].shape))

            vos_images = vos_data['images'].to(self._device)
            vos_initseganno = vos_data['given_seganno'].to(self._device)
            vos_segannos = vos_data['segannos'].to(self._device)

            if mode == 'train':
                self._optimizer.zero_grad()

            # Model inference
            vos_out, _ = self._model(vos_images, vos_initseganno, None)

            if len(vos_out['logsegs']) != 1:
                B,_,H,W = vos_initseganno.size()
                raise NotImplementedError("Model seems to track multiple targets during training, ids {}".format(vos_out['logsegs'].keys()))

            # Calculate total loss
            loss = {}
            B,N,K,H,W = vos_out['logsegs'][1].size()
            # print ("logsegs Size : " + str(vos_out['logsegs'][1].size()))
            # print ("vos_segannos Size : " + str(vos_segannos.size()))
            loss['vos'] = self.loss_W[0] * self._objective_vos(vos_out['logsegs'][1].view(B * N, K, H, W),
                                                                 vos_segannos.view(B * N, H, W))

            if self.DiffLoss and mode =='train':
                tmp = (vos_segannos[:, 1:, :, :, :] == 1).float().clamp(1e-7, 1 - 1e-7)
                est_heat = vos_out['Diff_heat'][1]
                # b, f, c, h, w = tmp.size()
                _, f, c, h, w = est_heat.size()

                tmp = tmp.view(B * f, 1, H, W)
                tmp = torch.nn.Upsample(size=(h, w), mode='bilinear')(torch.cat([1 - tmp, tmp], dim=1)) # chnn: 1->2
                T = tmp.view(B, f, c, h, w)

                E_1 = vos_out['prev_seg_tensor']
                E_1 =  torch.nn.Upsample(size=(h, w), mode='bilinear')(E_1.view(B * f, c, H,W))
                E_1 = E_1.view(B, f, c, h, w)

                loss['Diff'] = self.loss_W[1] * self.heatmap_loss(est_heat, (T - E_1))

            if self.TmpLoss and mode =='train':
                B, N, K, HS, WS = vos_out['Tmp_result'][1].size()
                vos_small_annos = torch.nn.Upsample(size=(HS, WS), mode='nearest')\
                    (vos_segannos.view(B * N, 1, H, W).float())
                # print("B,N,K,HS,WS : {} {} {} {} {}".format(B,N,K,HS,WS) )
                # print("B*N,H,W : {} {} {} ".format(B*N,H,W) )
                loss['Tmp'] = self.loss_W[2] * self.Tmp_loss(
                    vos_out['Tmp_result'][1].view(B * N, K, HS, WS),vos_small_annos)

            total_loss = sum(loss.values())

            # Backpropagate
            if mode == 'train':
                total_loss.backward()
                self._optimizer.step()
                if self._lr_sched is not None and self._lr_step != "batch":
                    curr_lr = get_thislr(self._optimizer)
                    iter = i + self._epoch * len(loader_vos)
                    for i in range(len(curr_lr)):
                        self._tb_logger.scalar_summary('lr'+str(i), curr_lr[i], iter)

                    curr_lr = self._lr_sched.get_lr(iter)
                    for param_group,w in zip(self._optimizer.param_groups,  self._args.Multi_W):
                        param_group['lr'] = curr_lr*w


            # Store vos loss and vos miou
            B,N,K,H,W = vos_out['logsegs'][1].size()
            loss['vos'] = loss['vos'].detach().to("cpu")
            self._stats[mode + '_vos loss'].update(loss['vos'].item(), B)
            vos_iou = utils.get_intersection_over_union(
                vos_out.get('logsegs')[1].view(B*N,K,H,W).detach(),
                vos_segannos.view(B*N,H,W).detach()).view(B, N, K) # Care only about channel 1 (target)
            tmp = vos_iou[:,:,1].mean(dim=1)
            if tmp.min() < vos_miou_extremes[0][0]: vos_miou_extremes[0] = (tmp.min(), i*B + tmp.argmin())
            if tmp.max() > vos_miou_extremes[1][0]: vos_miou_extremes[1] = (tmp.max(), i*B + tmp.argmax())

            vos_miou = vos_iou[:,:,1:].mean(dim=1).mean().to("cpu")
            self._stats[mode + '_vos mIoU'].update(vos_miou.item(), B)

            if loss.get('Tmp') is not None:
                loss['Tmp'] = loss['Tmp'].detach().to("cpu")
                self._stats[mode + '_vos Tmp'].update(loss['Tmp'].item(), B)

            if loss.get('Diff') is not None:
                loss['Diff'] = loss['Diff'].detach().to("cpu")
                self._stats[mode + '_vos Diff'].update(loss['Diff'].item(), B)

            # Save some statistics
            if (i + 1) % self._print_interval == 0:
                if self._stats.get(mode+'_vos loss') is not None:
                    logger.info("[{}: {}, {:5d}] Loss: {:.5f}".format(
                        mode, self._epoch,i+1, self._stats[mode+'_vos loss'].avg))

        # end for

        logger.info("[{}: {}] Loss vos: {:.5f} Loss Tmp: {:.5f} Loss Diff: {:.5f}".
                    format(mode, self._epoch, self._stats[mode+'_vos loss'].avg, self._stats[mode+'_vos Tmp'].avg,
                           self._stats[mode+'_vos Diff'].avg))
        logger.info("[{}: {}] mIoU vos: {:.5f}".format(mode, self._epoch, self._stats[mode+'_vos mIoU'].avg))
        logger.info("Worst mIoU this batch was {:.3f} (idx {}), and best {:.3f} (idx {})".format(vos_miou_extremes[0][0],
                                                                                           vos_miou_extremes[0][1],
                                                                                           vos_miou_extremes[1][0],
                                                                                           vos_miou_extremes[1][1]))


    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                            for seganno in video_part['given_segannos']]
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames


    def save_checkpoint(self, alternative_name=None):
        """Saves a checkpoint of the network and other variables."""

        net_type = type(self._model).__name__
        if torch.cuda.device_count() >1:
            save_dict = self._model.module.state_dict()
        else:
            save_dict = self._model.state_dict()
        curr_lr = get_thislr(self._optimizer)
        state = {
            'epoch': self._epoch,
            'net_type': net_type,
            'net': save_dict,
            'optimizer' : self._optimizer.state_dict(),
            'lr' : curr_lr,
            'stats' : self._stats,
            'use_gpu' : self._use_gpu,
            'best2017' : self.best2017,
            'best2016': self.best2016,
            'bestVal': self.bestVal,
            'ep2017': self.ep2017,
            'ep2016': self.ep2016,
            'epVal': self.epVal,

        }

        if alternative_name is not None:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, self._save_name + alternative_name, self._epoch)
        elif self._save_name is None:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, net_type, self._epoch)
        else:
            file_path = '{}/{}_ep{:04d}.pth.tar'.format(self._workspace_dir, self._save_name, self._epoch)
        print(file_path)
        self.prefix = file_path.split("_ep")[0]
        print(self.prefix)
        torch.save(state, file_path)


    def load_checkpoint(self, checkpoint_path = "", verbose=True):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        net_type = type(self._model).__name__

        checkpoint_dict = torch.load(checkpoint_path, map_location=self._device)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        self._epoch = checkpoint_dict['epoch'] + 1
        self._model.load_state_dict(checkpoint_dict['net'])
        self._optimizer.load_state_dict(checkpoint_dict['optimizer'])
        self._stats = checkpoint_dict['stats']
        self._use_gpu = checkpoint_dict['use_gpu']
        self.best2017, self.ep2017 = checkpoint_dict['best2017'], checkpoint_dict['ep2017']
        self.best2016, self.ep2016 = checkpoint_dict['best2016'], checkpoint_dict['ep2016']
        self.bestVal, self.epVal = checkpoint_dict['bestVal'], checkpoint_dict['epVal']

        curr_lr = checkpoint_dict['lr']

        for param_group, lr in zip(self._optimizer.param_groups, curr_lr):
            param_group['lr'] = lr

        if self._lr_step == "batch" and  self._lr_sched!=None:
            self._lr_sched.last_epoch= self._epoch
            self._lr_sched.step()


        if verbose:
            print("Loaded: {}".format(checkpoint_path))
