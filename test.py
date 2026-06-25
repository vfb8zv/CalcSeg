import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import DataLoader
from dataloader import DataLoading_3D
from tqdm import tqdm
from v3model import CalcSeg


##### utils #####

def load_matching_weights(model, checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint: 
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

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()

def myo_dice(y_true_f, y_pred_f, myo_mask):
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

def acti_pred(pred):
    act= nn.Sigmoid().to(DEVICE)
    return act(pred)

def plot_images(img1, gt,pred, alpha=0.5, cmap='gray', seg_cmap='autumn', loop_=0):
    os.makedirs('./results/plots/', exist_ok=True)

    fig, axes = plt.subplots(len(img1), 3, figsize=(5, 2*len(img1)))
    
    for i in range(len(img1)):
        axes[i][0].imshow(img1[i], cmap=cmap)
        axes[i][0].set_title("Raw")
        axes[i][0].axis("off")

        # Second image
        axes[i][1].imshow(img1[i], cmap=cmap)
        axes[i][1].imshow(gt[i], cmap=seg_cmap, alpha=alpha)
        axes[i][1].set_title("GT LGE")
        axes[i][1].axis("off")

        # Overlay
        axes[i][2].imshow(img1[i], cmap=cmap)
        axes[i][2].imshow(pred[i], cmap=seg_cmap, alpha=alpha) 
        axes[i][2].set_title("Pred LGE")
        axes[i][2].axis("off")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f'./results/plots/testplot_{loop_}.png', dpi=100)
            
##### dataloader #####

def get_data(data_dir="./dataset_cropped/", test_flag=False, batch_size=2, image_size=64, shuffle=True, subs=None):
    ds = DataLoading_3D(data_dir, test_flag=test_flag, image_size=image_size, subs=subs)
    datal = DataLoader(ds, batch_size=batch_size)
    return datal

datal= get_data('./dataset_cropped/', test_flag=True, batch_size=1, image_size=224)
d_= iter(datal)

try:
    low_subjects = np.loadtxt('./dataset_cropped/hard_subs.txt', dtype=str).tolist()

except:
    print('No clinical ground truth available.')
    low_subjects=[]


##### model defined #####

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = CalcSeg().to(DEVICE)
model=load_matching_weights(model, f'./results/models_curr_1/model_best.pt')
model.eval();
# model = enable_dropout(model)  ##  mc sampling only


##### testing #####

slices=[]
gt=[]
subjects=[]
dices=[]; hard_dices=[]; easy_dices=[]
burdens=[]
mc_samples=[]
pred=[]
hard_bd=[]

for _ in tqdm(range(len(datal)), desc="Subject"):
    
    x, y, label, mask, subject = next(d_)

    B,F,C,H,W = x.shape
    x = x.reshape(B*F, C, H, W)
    y = y.reshape(B*F, C, H, W)

    y= torch.nan_to_num(y, nan=0.0)

    mask_flat = mask.reshape(B * F)
    valid_idx = mask_flat.nonzero(as_tuple=True)[0]

    y_valid = y[valid_idx]
    x_valid = x[valid_idx]

    myo_mask = x_valid.clone()
    myo_mask[myo_mask>0]=1
    
    out = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=F)

    ## uncomment for MC sampling
    
    # out_mc = []
    # with torch.no_grad():
    #     for _ in range(100):
    #         y_hat = model.forward(x.to(DEVICE), mask = mask.to(DEVICE), Fr=12).detach().cpu()
    #         y_hat = acti_pred(y_hat[valid_idx])
    #         assert y_hat.shape==myo_mask.shape
    #         y_hat*=myo_mask
    #         out_mc.append(y_hat)
    # out_mc= torch.std(torch.stack(out_mc, dim=1), dim=1)
    

    out_valid = out[valid_idx]
    out_valid= acti_pred(out_valid).detach().cpu()*myo_mask
    y_valid*=myo_mask
        
    out_valid[out_valid<0.5]=0
    out_valid[out_valid>=0.5]=1

    full_dsc= [myo_dice(y_valid[p].unsqueeze(0), out_valid[p].unsqueeze(0), myo_mask[p].unsqueeze(0)) for p in range(len(y_valid))] # if torch.sum(y_valid[p])!=0]  ## for scar slices
    sb_error = [scar_burden_error(y_valid[p].unsqueeze(0), out_valid[p].unsqueeze(0).detach().cpu(), myo_mask[p].unsqueeze(0)) for p in range(len(y_valid))] # if torch.sum(y_valid[p])!=0] ## for scar slices


    if subject[0].strip() in [i.strip() for i in low_subjects]:
        # print(slice_files)
        full_dsc_hard = [myo_dice(y_valid[p].unsqueeze(0), out_valid[p].unsqueeze(0), myo_mask[p].unsqueeze(0)) for p in range(len(y_valid))]# if torch.sum(y_valid[p])!=0]     ## for scar slices
        hard_dices.append(torch.mean(torch.stack(full_dsc_hard, dim=0)))

        full_bd_hard = [scar_burden_error(y_valid[p].unsqueeze(0), out_valid[p].unsqueeze(0).detach().cpu(), myo_mask[p].unsqueeze(0).detach().cpu()) 
                        for p in range(len(y_valid)) ] #if torch.sum(y_valid[p])!=0]   ## for scar slices
        hard_bd.append(torch.mean(torch.stack(full_bd_hard, dim=0)))

    else:
        full_dsc_hard=[]

    dices.append(torch.mean(torch.stack(full_dsc, dim=0)))
    burdens.append(torch.mean(torch.stack(sb_error, dim=0)))

    # mc_samples.extend([out_mc.squeeze()[p] for p in range(len(out_mc)) if torch.sum(y_valid[p])!=0])  ## plots
    # mc_samples.append(out_mc.mean())                                                                  ## mean uncertainty
    
    ## plot predictions 
    if len(x_valid.squeeze())>1:
        slices.extend([x_valid.squeeze()[p] for p in range(len(x_valid.squeeze())) if torch.sum(y_valid.squeeze()[p])!=0])
        
        y_gt_viz = y_valid.detach().cpu().squeeze().clone()
        if len(y_gt_viz.shape)==2:
            y_gt_viz = y_gt_viz.unsqueeze(0)
        y_gt_viz[y_gt_viz==0]=np.nan
        gt.extend([y_gt_viz.squeeze()[p] for p in range(len(y_gt_viz.squeeze())) if torch.sum(y_valid.squeeze()[p])!=0])
        
        y_viz = out_valid.detach().cpu().squeeze().clone()
        if len(y_viz.shape)==2:
            y_viz = y_viz.unsqueeze(0)
        y_viz[y_viz==0]=np.nan
        pred.extend([y_viz.squeeze()[p] for p in range(len(y_viz.squeeze())) if torch.sum(y_valid.squeeze()[p])!=0])
        
    else:

        slices.extend(x_valid.squeeze())
        y_gt_viz = y_valid.detach().cpu().squeeze().clone()
        if len(y_gt_viz.shape)==2:
            y_gt_viz = y_gt_viz.unsqueeze(0)
        y_gt_viz[y_gt_viz==0]=np.nan
        gt.extend(y_gt_viz)
        
        y_viz = out_valid.detach().cpu().squeeze().clone()
        if len(y_viz.shape)==2:
            y_viz = y_viz.unsqueeze(0)
        y_viz[y_viz==0]=np.nan
        pred.extend(y_viz)


print('All subjects:')
print('Standard Deviation: ', float(torch.std(torch.tensor(dices))), '\n',
      'Median Dice: ', float(torch.median(torch.tensor(dices))), '\n',
      'Mediac Scar Burden Error', float(np.median(burdens))
     )
print()

if len(low_subjects)>0:
    print('Hard subjects:')
    print('Standard Deviation: ', float(torch.std(torch.tensor(hard_dices))), '\n',
        'Median Dice: ', float(torch.median(torch.tensor(hard_dices))), '\n',
        'Mediac Scar Burden Error', float(np.median(hard_bd))
        ) 
    print()

idix = 20  ## change for plot
sll= 5     ## change for plot
loops = 2  ## change for plot

for loop_ in range(loops):
    plot_images(slices[idix:idix+sll], gt[idix:idix+sll], pred[idix:idix+sll], alpha=0.8, loop_=loop_)
    idix+=sll