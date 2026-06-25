import numpy as np
from torch import nn
import os
import torch
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
from dataloader import DataLoading_3D
from v3model import CalcSeg
import math


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def load_matching_weights(model, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:  # handle Lightning or custom checkpoints
        checkpoint = checkpoint["state_dict"]

    model_dict = model.state_dict()

    matched_weights = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(matched_weights)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched_weights)}/{len(model_dict)} layers from checkpoint")
    return model

def acti_pred(pred):
    act= nn.Sigmoid().to(DEVICE)
    return act(pred)


def myo_dice(y_true_f, y_pred_f, myo_mask):
    y_pred_f = acti_pred(y_pred_f)
    fg_mask = myo_mask == 1
    y_true_f = y_true_f[fg_mask].view(-1)
    y_pred_f = y_pred_f[fg_mask].view(-1)
    return ((2 * (y_true_f * y_pred_f).sum()) / (y_true_f.sum() + y_pred_f.sum() + 1e-14))

def scar_burden_error(Y, Y_hat, M):
    Y_myo = Y[M == 1]
    Yhat_myo = Y_hat[M == 1]
    gt_scar_vol = torch.sum(Y_myo)
    pred_scar_vol = torch.sum(Yhat_myo)
    scar_burden_error = abs(gt_scar_vol - pred_scar_vol) / (gt_scar_vol + 1e-14) * 100
    return scar_burden_error

######### data loader ###########


def get_data(data_dir="./dataset_cropped/", test_flag=False, batch_size=2, image_size=64, shuffle=True, subs=None):
    ds = DataLoading_3D(data_dir, test_flag=test_flag, image_size=image_size, subs=subs)
    datal = DataLoader(ds)
    return datal

############### model utils ###################

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def model_main(args):
    if args.stage>0:
        subs = np.loadtxt(f'./subs/hard_{args.stage-1}.txt', dtype=str).tolist()
        subs_easy = np.loadtxt(f'./subs/other_{args.stage-1}.txt', dtype=str).tolist()
        print(f"Loaded {len(subs)} hard subjects from previous stage.")
    else:
        subs=None
    datal= get_data('./dataset_cropped/', test_flag=False, batch_size=1, image_size=224, subs=subs)
    d_= iter(datal)
    print('Training steps per epoch: ',len(datal))

    os.makedirs('./subs', exist_ok=True)
    
    model = CalcSeg().to(DEVICE)
    model=load_matching_weights(model, f'./results/models_{args.loss_type}_{args.stage}/model_best.pt')
    
    model.eval()
    enable_dropout(model) 
    
    loss_pre= nn.Sigmoid()
    
    d_= iter(datal)
    easy_subjects=[]; hard_subjects=[]
    
    #### STEPS ####
    Di_dsc=[]; Di_sbe=[]; Di_unc=[]; Di_subs=[]
    for _ in range(len(datal)):
        try:
            x, y, labels, mask, subject = next(d_)
        except:
            d_ = iter(datal)
            x, y, labels, mask, subject = next(d_)
            
        B,F,C,H,W = x.shape
        x = x.reshape(B*F, C, H, W)
        y = y.reshape(B*F, C, H, W)
        
        if y.sum()==0:
            continue
        
        with torch.no_grad():
            out = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=F)
        
        y= torch.nan_to_num(y, nan=0.0)
    
        mask_flat = mask.reshape(B * F)
        valid_idx = mask_flat.nonzero(as_tuple=True)[0]
        out_valid = acti_pred(out[valid_idx])
        y_valid = y[valid_idx]
        myo_mask = x[valid_idx].clone()
        myo_mask[myo_mask>0]=1

        out_mc = []
        with torch.no_grad():
            for _ in range(args.n_mc_samples): 
                y_hat = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=F)
                out_mc.append(acti_pred(y_hat[valid_idx]))
        out_mc= torch.std(torch.stack(out_mc, dim=1), dim=1)
        out_mc = out_mc.detach().cpu()
        
        Di_dsc.append(float(myo_dice(y_valid.to(DEVICE), out_valid, myo_mask.to(DEVICE))))
        Di_sbe.append(float(scar_burden_error(y_valid.to(DEVICE), out_valid, myo_mask.to(DEVICE))))
        Di_unc.append(float(out_mc.mean()))
        Di_subs.append(subject[0])
        
    dice_thresh = torch.median(torch.tensor(Di_dsc))
    sbe_thresh = torch.median(torch.tensor(Di_sbe))
    unc_thresh = torch.median(torch.tensor(Di_unc))

    oom_d = min(1.0, 10 ** -(math.floor(math.log10(dice_thresh+1e-8))+1))
    oom_be = 10 ** -(math.floor(math.log10(sbe_thresh+1e-8))+1)
    oom_uc = min(1e-1, 10 ** -(math.floor(math.log10(max(unc_thresh + 1e-8, 1e-12)))+1))
    gammas=[oom_d, oom_be, oom_uc]
    print(gammas, dice_thresh, sbe_thresh, unc_thresh)

    for i in range(len(Di_subs)): 
        print(gammas[0]*Di_dsc[i], gammas[1]*Di_sbe[i], gammas[2]*Di_unc[i])
        if (gammas[0]*Di_dsc[i]<dice_thresh) or gammas[1]*Di_sbe[i]>sbe_thresh or gammas[2]*Di_unc[i]> unc_thresh:
            hard_subjects.append(Di_subs[i])
        else:
            easy_subjects.append(Di_subs[i])
            
    easy_subjects = list(set(easy_subjects)) + subs_easy
    hard_subjects = list(set(hard_subjects))

    with open(f"./subs/other_{args.stage}.txt", "w") as f:
        for s in easy_subjects:
            f.write(f"{s}\n")

    with open(f"./subs/hard_{args.stage}.txt", "w") as f:
        for s in hard_subjects:
            f.write(f"{s}\n")

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default = 1, help="Batch size")
    parser.add_argument("--stage", type=int, default = 0, help="Curriculum Stage")
    parser.add_argument("--n_mc_samples", type=int, default = 5, help="Total no. of MC samples")
    parser.add_argument("--loss_type", type=str,  default='pre', help="Curriculum model vs. pretraining")
    args = parser.parse_args()
    
    model_main(args)

if __name__ == "__main__":
    main()
