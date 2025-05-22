import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from .training_tools import AverageMeter, MetricComputer

@torch.no_grad()
def eval_model(args, posthoc_layer, loader, num_classes):
    epoch_summary = {"Accuracy": AverageMeter()}
    tqdm_loader = tqdm(loader)
    computer = MetricComputer(n_classes=num_classes)
    all_preds = []
    all_labels = []
    
    for batch_X, batch_Y in tqdm(loader):
        batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device) 
        out = posthoc_layer(batch_X)            
        all_preds.append(out.detach().cpu().numpy())
        all_labels.append(batch_Y.detach().cpu().numpy())
        metrics = computer(out, batch_Y) 
        epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0]) 
        summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
        summary_text = "Eval - " + " ".join(summary_text)
        tqdm_loader.set_description(summary_text)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_labels.max() == 1:
        auc = roc_auc_score(all_labels, softmax(all_preds, axis=1)[:, 1])
        return auc
    return epoch_summary["Accuracy"]

def train_hybrid(args, train_loader, val_loader, posthoc_layer, optimizer, num_classes):
    cls_criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.num_epochs+1):
        print(f"Epoch: {epoch}")
        epoch_summary = {"CELoss": AverageMeter(),
                         "Accuracy": AverageMeter()}
        tqdm_loader = tqdm(train_loader)
        computer = MetricComputer(n_classes=num_classes)
        for batch_X, batch_Y in tqdm(train_loader):
            batch_X, batch_Y = batch_X.to(args.device), batch_Y.to(args.device)
            optimizer.zero_grad()
            out, projections = posthoc_layer(batch_X, return_dist=True)
            cls_loss = cls_criterion(out, batch_Y)
            loss = cls_loss + args.l2_penalty*(posthoc_layer.residual_classifier.weight**2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_summary["CELoss"].update(cls_loss.detach().item(), batch_X.shape[0])
            metrics = computer(out, batch_Y) 
            epoch_summary["Accuracy"].update(metrics["accuracy"], batch_X.shape[0])

            summary_text = [f"Avg. {k}: {v.avg:.3f}" for k, v in epoch_summary.items()]
            summary_text = " ".join(summary_text)
            tqdm_loader.set_description(summary_text)
        
        latest_info = dict()
        latest_info["epoch"] = epoch
        latest_info["args"] = args
        latest_info["train_acc"] = epoch_summary["Accuracy"]
        latest_info["test_acc"] = eval_model(args, posthoc_layer, val_loader, num_classes)
        print("Final test acc: ", latest_info["test_acc"])
    return latest_info
