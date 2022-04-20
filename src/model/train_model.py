from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model.model import *


def train_model(model, optimizer, 
                train_loader, test_loader,
                num_epochs, writer=None, 
                gradient_type='normal', gradient_clip=1.0,
                computeMSE=False, verbose=True, 
                save_folder=None, save_every = 20, 
                device = 'cuda'):
    
    model.to(device)
    model.train()
    train_losses = {}
    eval_losses = {}
    batch_ct = 0
    print(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in tqdm(range(num_epochs)):
        
        # ---TRAINING---
        for idx, (inputs, _) in enumerate(train_loader):
            
            inputs = inputs.to(device)
            optimizer.zero_grad()
            _, _, loss = model.loss(inputs, computeMSE = computeMSE)
            loss.backward()
            
            # Clip gradient depending on type
            if gradient_clip:
                if gradient_type == "normal":
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip, norm_type=2)
                elif gradient_type == "value":
                    nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clip)
            
            optimizer.step()
            
            # Save loss
            for k, v in model.result_dict.items():
                train_losses.setdefault(k, []).append(v)
                if writer: # write to tensorboard 
                    writer.add_scalar(f"Loss/train/{k}", v, batch_ct)
                
        if save_folder:
            if (epoch+1) % save_every == 0 or epoch == num_epochs-1:
                torch.save(model.state_dict(), f"{save_folder}/model_E{epoch+1}")
             
        # ---VALIDATION---
        model.eval()
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                _, _, loss = model.loss(inputs, computeMSE = computeMSE)
                
                # Save loss
                for k, v in model.result_dict.items():
                    eval_losses.setdefault(k, []).append(v)
                    if writer: # write to tensorboard
                        writer.add_scalar(f"Loss/eval/{k}", v, batch_ct)
        batch_ct += 1
            
    return train_losses, eval_losses