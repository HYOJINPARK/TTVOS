import os
import time
import torch
import utils
import logging
import cv2

logger = logging.getLogger('global')

def TensorToImg(torch_tensor):
    if len(torch_tensor.size())>2:
        torch_tensor = torch.mean(torch_tensor,dim=0)
    min_v = torch.min(torch_tensor)
    max_v = torch.max(torch_tensor)
    out = 255*(torch_tensor-min_v)/(max_v-min_v)
    return out

def TensorToGray(torch_tensor):
    if len(torch_tensor.size())>2:
        torch_tensor = torch.nn.Softmax(dim=0)(torch_tensor)
        torch_tensor=torch_tensor[1,:,:]

    out = 255*torch_tensor
    return  out


class VOSEvaluator(object):
    def __init__(self, dataset, device='cuda', save=False, debug= False):
        self._dataset = dataset
        self._device = device
        self._save = save
        self._imsavehlp = utils.ImageSaveHelper()
        self.debug = debug
        if dataset.__class__.__name__ == 'DAVIS17V2':
            self._sdm = utils.ReadSaveDAVISChallengeLabels()
        else:
            self._sdm = utils.ReadSaveYTBChallengeLabels()
    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                          for seganno in video_part['given_segannos']]
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames

    def evaluate_video(self, model, seqname, video_parts, output_path, save):
        # target_f = ["21198a32d0", "c84a452ec6"] #0b97736357", "37dc952545" ]
        # print(seqname)
        for video_part in video_parts:
            # if seqname in target_f :
            #     print("hope to find")
            images, given_segannos, _, fnames = self.read_video_part(video_part)


            #####################fix later###############

            with torch.no_grad():


                # _, _, _, orgH, orgW = images.size()
                # Temp = torch.nn.Upsample(size=(240, 432), mode='bilinear')(images[0])
                # images = Temp.unsqueeze(0)
                # for i in range(len(given_segannos)):
                #     if given_segannos[i] != None:
                #         given_segannos[i] = torch.nn.Upsample(size=(240, 432), mode='nearest')(given_segannos[i].float() ).long()

                t0 = time.time()
                if self.debug:
                    tracker_out, state = model(images, given_segannos, None, vis_res=self.debug)
                else:
                    tracker_out, _ = model(images, given_segannos, None)

                t1 = time.time()
                # Temp=  torch.nn.Upsample(size=(orgH, orgW), mode='nearest')(tracker_out['segs'][0].float()).long()
                # tracker_out['segs']=Temp.unsqueeze(0)

            if save is True:
                if self.debug:
                    Obj_info = tracker_out['segs_vis']
                for idx in range(len(fnames)):
                    fpath = os.path.join(output_path, seqname, fnames[idx])
                    data = ((tracker_out['segs'][0, idx, 0, :, :].cpu().byte().numpy(), fpath), self._sdm)
                    self._imsavehlp.enqueue(data)
                    if self.debug and idx >0:
                        file_name = fpath.split(".png")[0]

                        for k, v in Obj_info.items():
                            visual_Gtensor, visual_Gname, visual_Itensor, visual_Iname, visual_tensor, visual_name = v[idx-1]

                            if visual_Gtensor != None:
                                for ThisTensor, ThisName in zip(visual_Gtensor, visual_Gname):
                                    This_vis = TensorToGray(ThisTensor[0]).cpu().byte().numpy()
                                    cv2.imwrite(file_name + ThisName + "_" + str(k)+ ".png", This_vis)
                            if visual_Itensor != None:
                                for ThisTensor, ThisName in zip(visual_Itensor, visual_Iname):
                                    This_vis = TensorToImg(ThisTensor[0]).cpu().byte().numpy()
                                    cv2.imwrite(file_name + ThisName + "_" + str(k) + ".png", This_vis)
                            if visual_tensor != None:
                                for ThisTensor, ThisName in zip(visual_tensor, visual_name):
                                    This_vis = (255 * ThisTensor[0]).cpu().byte().numpy()
                                    cv2.imwrite(file_name + ThisName + "_" + str(k)+  ".png", This_vis)


                        # visual_Gtensor, visual_Gname, visual_Itensor, visual_Iname,\
                        #     visual_tensor, visual_name = Obj1_info[1][idx-1]
                        #
                        # if visual_Gtensor !=None:
                        #     for ThisTensor, ThisName in zip(visual_Gtensor, visual_Gname):
                        #         This_vis = TensorToGray(ThisTensor[0]).cpu().byte().numpy()
                        #         cv2.imwrite(file_name+ThisName+".png", This_vis)
                        # if visual_Itensor !=None:
                        #     for ThisTensor, ThisName in zip(visual_Itensor, visual_Iname):
                        #         This_vis = TensorToImg(ThisTensor[0]).cpu().byte().numpy()
                        #         cv2.imwrite(file_name+ThisName+".png", This_vis)
                        # if visual_tensor !=None:
                        #     for ThisTensor, ThisName in zip(visual_tensor, visual_name):
                        #         This_vis = (255*ThisTensor[0]).cpu().byte().numpy()
                        #         cv2.imwrite(file_name+ThisName+".png", This_vis)
                        #


        return t1-t0, len(fnames)

    def evaluate(self, model, output_path):
        model.to(self._device)
        model.eval()
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        with torch.no_grad():
            tot_time, tot_frames, video_seq = 0.0, 0.0, 0
            for seqname, video_parts in self._dataset.get_video_generator(test=True):
                savepath = os.path.join(output_path, seqname)
                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                time_elapsed, frames = self.evaluate_video(model, seqname, video_parts, output_path, self._save)


                tot_time += time_elapsed
                tot_frames += frames
                video_seq += 1

                # if self._save is False:
                #     logger.info(seqname + 'FPS:{}, frames:{}, time:{}'.format(frames / time_elapsed, frames, time_elapsed))
                # else:
                #     logger.info(seqname + '\tsaved')

            logger.info('Total seq saved')
            logger.info('Total FPS:{}\n\n'.format(tot_frames/tot_time))

        self._imsavehlp.kill()
