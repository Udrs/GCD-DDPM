

import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
from torchvision.utils import save_image
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset, BRATSDataset3D
from guided_diffusion.isicloader import ISICDataset
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name == 'ISIC':
        tran_list = [transforms.Resize((args.image_size,args.image_size)), transforms.ToTensor(),]
        transform_test = transforms.Compose(tran_list)

        ds = ISICDataset(args, args.data_dir, transform_test)
        args.in_ch = 4
    elif args.data_name == 'BRATS':
        tran_list = [transforms.Resize((args.image_size,args.image_size)),]
        transform_test = transforms.Compose(tran_list)

        ds = BRATSDataset3D(args.data_dir,transform_test)
        args.in_ch = 5
    datal = th.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True)
    data = iter(datal)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    all_images = []

    model_path = "/media/xxx/savedmodel465000.pt"
    state_dict = dist_util.load_state_dict(model_path, map_location="cpu")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # name = k[7:] # remove `module.`
        if 'module.' in k:
            new_state_dict[k[7:]] = v
            # load params
        else:
            new_state_dict = state_dict

    model.load_state_dict(new_state_dict)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    time_step = 0
    while len(all_images) * args.batch_size < args.num_samples:
        A_img, B_img, CD_img, name = next(data)  #should return an image from the dataloader "data"
        c = th.randn_like(A_img[:, :1, ...])
        sum = th.cat((A_img,B_img),dim = 1)
        img = th.cat((sum, c), dim=1)     #add a noise channel$
        # if args.data_name == 'ISIC':
        #     slice_ID=path[0].split("_")[-1].split('.')[0]
        # elif args.data_name == 'BRATS':
        #     # slice_ID=path[0].split("_")[2] + "_" + path[0].split("_")[4]
        #     slice_ID=path[0].split("_")[-3] + "_" + path[0].split("slice")[-1].split('.nii')[0]
        time_step = time_step + 1
        logger.log("sampling...")

        start = th.cuda.Event(enable_timing=True)
        end = th.cuda.Event(enable_timing=True)
        enslist = []

        for i in range(5):  #this is for the generation of an ensemble of 5 masks.
            model_kwargs = {}
            start.record()
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            sample, x_noisy, org, cal, cal_out = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size), img,
                step = args.diffusion_steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            end.record()
            th.cuda.synchronize()
            print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
     
            co = th.tensor(cal_out)
            enslist.append(co)

            # if args.debug:
            #     # print('sample size is',sample.size())
            #     # print('org size is',org.size())
            #     # print('cal size is',cal.size())
            #     if args.data_name == 'ISIC':
            #         s = th.tensor(sample)[:,-1,:,:].unsqueeze(1).repeat(1, 3, 1, 1)
            #         o = th.tensor(org)[:,:-1,:,:]
            #         c = th.tensor(cal).repeat(1, 3, 1, 1)
            #         co = co.repeat(1, 3, 1, 1)
            #     elif args.data_name == 'BRATS':
            #         s = th.tensor(sample)[:,-1,:,:].unsqueeze(1)
            #         m = th.tensor(m.to(device = 'cuda:0'))[:,0,:,:].unsqueeze(1)
            #         o1 = th.tensor(org)[:,0,:,:].unsqueeze(1)
            #         o2 = th.tensor(org)[:,1,:,:].unsqueeze(1)
            #         o3 = th.tensor(org)[:,2,:,:].unsqueeze(1)
            #         o4 = th.tensor(org)[:,3,:,:].unsqueeze(1)
            #         c = th.tensor(cal)

            #     tup = (o1/o1.max(),o2/o2.max(),o3/o3.max(),o4/o4.max(),m,s,c,co)

            #     compose = th.cat(tup,0)
            #     vutils.save_image(compose, fp = args.out_dir +'_output'+str(i)+".jpg", nrow = 1, padding = 10)
        ensres = staple(th.stack(enslist,dim=0)).squeeze(0)
    
def create_argparser():
    defaults = dict(
        data_name = 'ISIC',
        data_dir="../dataset/brats2020/testing",
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        model_path="",
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = "0",
        out_dir='./results_3_10_test/',
        multi_gpu = None, #"0,1,2"
        debug = False
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":

    main()
