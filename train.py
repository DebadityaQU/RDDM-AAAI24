import torch
import wandb
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from model import DiffusionUNetCrossAttention, ConditionNet
from diffusion import RDDM
from data import get_datasets
import torch.nn as nn
from metrics import *
from lr_scheduler import CosineAnnealingLRWarmup
from torch.utils.data import Dataset, DataLoader

def unfreeze_model(model):

    for param in model.parameters():
        param.requires_grad = True

    model.train()
    
    return model

def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

set_deterministic(31)

def train_rddm(config):

    n_epoch = config["n_epoch"]
    device = config["device"]
    batch_size = config["batch_size"]
    nT = config["nT"]
    num_heads = config["attention_heads"]
    cond_mask = config["cond_mask"]
    alpha1 = config["alpha1"]
    alpha2 = config["alpha2"]
    PATH = config["PATH"]

    rddm = RDDM(
        eps_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        region_model=DiffusionUNetCrossAttention(512, 1, device, num_heads=num_heads),
        betas=(1e-4, 0.2), 
        n_T=nT
    )

    Conditioning_network1 = ConditionNet().to(device)
    Conditioning_network2 = ConditionNet().to(device)
    rddm.to(device)

    optim = torch.optim.AdamW([*rddm.parameters(), *Conditioning_network1.parameters(), *Conditioning_network2.parameters()], lr=1e-5)

    if torch.cuda.device_count() > 1:
        rddm = nn.DataParallel(rddm)
        Conditioning_network1 = nn.DataParallel(Conditioning_network1)
        Conditioning_network2 = nn.DataParallel(Conditioning_network2)

    scheduler = CosineAnnealingLRWarmup(optim, T_max=150, T_warmup=20)
    
    #传入参数data_type 用于指定使用陀螺仪数据还是加速度计的数据
    dataset_train, dataset_val, _ = get_datasets(source='bcg',dest='ppg')

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=64)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=64)
    
    # Early stopping 参数
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for i in range(n_epoch):
        print(f"\n****************** Epoch - {i} *******************\n\n")

        unfreeze_model(rddm)
        unfreeze_model(Conditioning_network1)
        unfreeze_model(Conditioning_network2)
        
        train_loss = 0.0
        
        pbar = tqdm(train_loader)
        optim.zero_grad()
        
        for y_ecg, x_ppg, ecg_roi in pbar:
            
            ## Train Diffusion
            optim.zero_grad()
            x_ppg = x_ppg.float().to(device)
            y_ecg = y_ecg.float().to(device)
            ecg_roi = ecg_roi.float().to(device)

            ppg_conditions1 = Conditioning_network1(x_ppg, drop_prob=cond_mask)
            ppg_conditions2 = Conditioning_network2(x_ppg, drop_prob=cond_mask)

            ddpm_loss, region_loss = rddm(x=y_ecg, cond1=ppg_conditions1, cond2=ppg_conditions2, patch_labels=ecg_roi)

            ddpm_loss = alpha1 * ddpm_loss
            region_loss = alpha2 * region_loss
            
            loss = ddpm_loss + region_loss

            loss.mean().backward()
            
            optim.step()

            pbar.set_description(f"loss: {loss.mean().item():.4f}")
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        rddm.eval()
        Conditioning_network1.eval()
        Conditioning_network2.eval()    
        
        val_loss = 0.0
        with torch.no_grad():
            for y_ppg, x_acc, ppg_roi in val_loader:
                x_acc = x_acc.float().to(device)
                y_ppg = y_ppg.float().to(device)
                ppg_roi = ppg_roi.float().to(device)

                acc_conditions1 = Conditioning_network1(x_acc)
                acc_conditions2 = Conditioning_network2(x_acc)

                ddpm_loss, region_loss = rddm(x=y_ppg, cond1=acc_conditions1, cond2=acc_conditions2, patch_labels=ppg_roi)

                ddpm_loss = alpha1 * ddpm_loss
                region_loss = alpha2 * region_loss

                loss = ddpm_loss + region_loss
                val_loss += loss.mean().item()

        val_loss /= len(val_loader)

        print(f"Epoch {i}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        wandb.log({
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "DDPM_loss": ddpm_loss.mean().item(),
            "Region_loss": region_loss.mean().item(),
        })

        # Early stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(rddm.state_dict(), f"{PATH}/RDDM_best.pth")
            torch.save(Conditioning_network1.state_dict(), f"{PATH}/ConditionNet1_best.pth")
            torch.save(Conditioning_network2.state_dict(), f"{PATH}/ConditionNet2_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {i+1} epochs")
            break
            
        scheduler.step()

                
if __name__ == "__main__":

    config = {
        "n_epoch": 150,
        "batch_size": 128*4,
        "nT":20,
        "device": "cuda",
        "attention_heads": 8,
        "cond_mask": 0.0,
        "alpha1": 100,
        "alpha2": 10,
        "PATH": "./models/"
    }

    wandb.login(key="02b9e33b3e30a62ef968f5b98302b3d8bb54be7d")
    wandb.init(
        project="RDDM",
        #entity="kim_tsai",
        id="test",
        config=config
    )
    
    train_rddm(config)