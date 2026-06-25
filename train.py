import scipy.io as sio
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import shutil, os, json
import torch
import random
from torch.utils.data import Dataset, Sampler, Subset, DataLoader
import argparse
from dataloader import DataLoading_3D
from scipy import ndimage
from sklearn.model_selection import train_test_split
import torchvision
import skimage
from v3model import CalcSeg
import torch.nn.functional as F


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


############### model utils ###################


def weighted_dice_loss(y_true, y_pred, myo_mask, w_fg=0.75, w_bg=0.25, smooth=1e-10):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    myo_mask = myo_mask.view(-1).to(DEVICE)
    fg_mask = myo_mask == 1
    y_true_f = y_true_f[fg_mask]
    y_pred_f = y_pred_f[fg_mask]
    intersection_fg = (y_true_f * y_pred_f).sum()
    dice_fg = (2 * intersection_fg + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)
    y_true_bg = 1 - y_true_f
    y_pred_bg = 1 - y_pred_f
    intersection_bg = (y_true_bg * y_pred_bg).sum()
    dice_bg = (2 * intersection_bg + smooth) / (y_true_bg.sum() + y_pred_bg.sum() + smooth)
    return 1 - (w_fg * dice_fg + w_bg * dice_bg)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def model_main(args):
    
    batch_size = args.batch_size
    epochs= args.epochs
    model_save_path= f'{args.loss_type}_{args.model_type}_{epochs}.pt'

    if 'curr' in args.loss_type:
        subs= np.loadtxt(f"./subs/other_{args.model_type-1}.txt", dtype=str).tolist()
    else:
        subs=None

    datal_full = DataLoading_3D('./dataset_cropped/', test_flag=False, image_size=224, subs=subs)
    labels_array = np.array(datal_full.labels)

    train_idx, val_idx = train_test_split(
            np.arange(len(labels_array)),
            test_size=0.1,
            stratify=labels_array,
            random_state=42)
        
    ds_train = Subset(datal_full, train_idx)
    ds_train.labels = labels_array[train_idx]
    datal = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    
    ds_val = Subset(datal_full, val_idx)
    ds_val.labels = labels_array[val_idx]
    datal_testing = DataLoader(ds_val, batch_size=4)
    
    d_= iter(datal)
    print('Training steps per epoch: ',len(datal))
    
    model = CalcSeg().to(DEVICE)
    loss_scale = 10
    
    try:
        model=load_matching_weights(model, f'./results/models_curr_{args.model_type-1}/model_best.pt')
        print('loaded pretrained weights')
    except:
        print('no pretrained model for segmentation')

    ################## Model Training ########################

    os.makedirs('./results/', exist_ok=True)
    os.makedirs(f'./results/models_{args.loss_type}_{args.model_type}/', exist_ok=True)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay= 1e-8, lr=1e-4)

    loss_pre= nn.Sigmoid()    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0

    for ep in range(epochs):
        losses=[]
        focal_losses, dice_losses = [], []
        val_losses=[]

        d_= iter(datal)

        for _ in range(len(datal)):
            try:
                x, y, labels, mask, subject= next(d_)
            except:
                d_ = iter(datal)
                x, y, labels, mask, subject= next(d_)
                
            B,F,C,H,W = x.shape
            x = x.reshape(B*F, C, H, W)
            y = y.reshape(B*F, C, H, W)
        
            optimizer.zero_grad()
            
            out = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=F)
            
            y= torch.nan_to_num(y, nan=0.0)
            mask_flat = mask.reshape(B * F)
            valid_idx = mask_flat.nonzero(as_tuple=True)[0]
            # print(valid_idx)
            out_valid = out[valid_idx]
            y_valid = y[valid_idx]
            myo_mask = x[valid_idx].clone()
            myo_mask[myo_mask>0]=1

            f_loss = loss_scale * torchvision.ops.sigmoid_focal_loss(out_valid, y_valid.to(DEVICE), reduction='mean')
            d_loss = weighted_dice_loss(loss_pre(out_valid), y_valid.to(DEVICE), myo_mask.to(DEVICE))

            loss= f_loss + d_loss 

            focal_losses.append(f_loss.mean().item())
            dice_losses.append(d_loss.mean().item())

            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        mean_focal = torch.mean(torch.tensor(focal_losses))
        mean_dice = torch.mean(torch.tensor(dice_losses))
        mean_total = torch.mean(torch.tensor(losses))

        print(f"Epoch {ep}: "
              f"focal={mean_focal:.5f}, "
              f"dice={mean_dice:.5f}, "
              f"total={mean_total:.5f}")

        if (ep+1)%1==0:
            d_test = iter(datal_testing)
            model.eval()
            with torch.no_grad():
                for _ in range(len(datal_testing)):
                    try:
                        x, y, labels, mask, subject = next(d_test)
                    except:
                        d_test = iter(datal_testing)
                        x, y, labels, mask, subject = next(d_test)
                        
                    B,F,C,H,W = x.shape
                    x = x.reshape(B*F, C, H, W)
                    y = y.reshape(B*F, C, H, W)
                    
                    out = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=F)
                    y= torch.nan_to_num(y, nan=0.0)

                    mask_flat = mask.reshape(B * F)
                    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
                    out_valid = out[valid_idx]
                    y_valid = y[valid_idx]
                    myo_mask = x[valid_idx].clone()
                    myo_mask[myo_mask>0]=1

                    f_loss = loss_scale * torchvision.ops.sigmoid_focal_loss(out_valid, y_valid.to(DEVICE), reduction='mean')
                    d_loss = weighted_dice_loss(loss_pre(out_valid), y_valid.to(DEVICE), myo_mask.to(DEVICE))

                    v_loss = float((d_loss +f_loss).item())
                    val_losses.append(v_loss)
        model.train()

        mean_val = torch.mean(torch.tensor(val_losses))
        print(f'Validation {ep}: {mean_val}')
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(model.state_dict(), f'./results/models_{args.loss_type}_{args.model_type}/model_best.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {ep} (patience={patience})')
                break
                
        if ep%25==0:
            torch.save(model.state_dict(), f'./results/models_{args.loss_type}_{args.model_type}/{ep}_'+model_save_path)
        torch.save(model.state_dict(), f'./results/models_{args.loss_type}_{args.model_type}/'+model_save_path)

    torch.save(model.state_dict(), f'./results/models_{args.loss_type}_{args.model_type}/'+model_save_path)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default = 2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=300, help="Training Epochs")
    parser.add_argument("--model_type", type=int,  default=0, help="Curriculum stage")
    parser.add_argument("--loss_type", type=str,  default='pre', help="Curriculum model vs. pretraining")
    args = parser.parse_args()
    model_main(args)

if __name__ == "__main__":
    main()