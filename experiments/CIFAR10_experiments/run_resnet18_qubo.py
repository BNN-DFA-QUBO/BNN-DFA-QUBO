"""Transfer-learning CIFAR-10 benchmark with five shared linear/binary heads.

This is a separate experimental track; it does NOT replace the BNN-DFA baseline.
Backbone: ImageNet-pretrained ResNet18 fine-tuned on CIFAR-10.
All heads consume the same frozen 512-D penultimate representation.
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

from experiments.CIFAR10_experiments.cifar_dfa_least_squares import fit_and_evaluate_ridge
from experiments.CIFAR10_experiments.cifar_dfa_binarized_ls import evaluate_binarized_head
from experiments.CIFAR10_experiments.cifar_dfa_direct_binary import evaluate_direct_binary_head
from experiments.CIFAR10_experiments.cifar_dfa_ste_binary import evaluate_ste_head
from experiments.CIFAR10_experiments.cifar_dfa_qubo import fit_fixed_qubo, evaluate_qubo
from experiments.CIFAR10_experiments.tune_qubo_validation import tune_from_representations

MEAN=(0.485,0.456,0.406)
STD=(0.229,0.224,0.225)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class ResNetFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        self.net=models.resnet18(weights=weights)
        self.net.fc=nn.Identity()
    def forward(self,x): return self.net(x)

def accuracy(logits,y):
    return (logits.argmax(1)==y).float().mean().item()*100

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=20)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--validation-fraction",type=float,default=.1)
    p.add_argument("--num-reads",type=int,default=50)
    p.add_argument("--num-sweeps",type=int,default=500)
    p.add_argument("--qubo-tune",action="store_true")
    p.add_argument("--data-dir",type=Path,default=Path("data/cifar10/raw"))
    p.add_argument("--output-dir",type=Path,default=Path("data/resnet18_cifar10"))
    p.add_argument("--no-pretrained",action="store_true")
    args=p.parse_args()
    if not 0<args.validation_fraction<1: p.error("validation-fraction must be between 0 and 1")
    seed_everything(args.seed)
    device=torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"ResNet18 CIFAR-10 | seed={args.seed} | device={device}")
    print("Pretrained ImageNet weights are downloaded on first run if not cached.")
    aug=transforms.Compose([transforms.Resize(96),transforms.RandomCrop(96,padding=8),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    ev=transforms.Compose([transforms.Resize(96),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    raw_train=datasets.CIFAR10(str(args.data_dir),train=True,download=True)
    raw_test=datasets.CIFAR10(str(args.data_dir),train=False,download=True)
    yall=np.asarray(raw_train.targets)
    from sklearn.model_selection import train_test_split
    tr,va=train_test_split(np.arange(len(raw_train)),test_size=args.validation_fraction,random_state=args.seed,stratify=yall)
    class View(torch.utils.data.Dataset):
        def __init__(self,base,indices,tf): self.base,self.indices,self.tf=base,indices,tf
        def __len__(self): return len(self.indices)
        def __getitem__(self,i):
            x,y=self.base[int(self.indices[i])]; return self.tf(x),y
    train_aug=View(raw_train,tr,aug); train_eval=View(raw_train,tr,ev)
    val_ds=View(raw_train,va,ev); test_ds=View(raw_test,np.arange(len(raw_test)),ev)
    workers=0
    train_loader=DataLoader(train_aug,batch_size=args.batch_size,shuffle=True,num_workers=workers)
    eval_loaders=[DataLoader(d,batch_size=args.batch_size*2,shuffle=False,num_workers=workers) for d in (train_eval,val_ds,test_ds)]
    model=models.resnet18(weights=None) if args.no_pretrained else models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc=nn.Linear(model.fc.in_features,10)
    model=model.to(device)
    # Fine-tune full network; low LR protects pretrained visual features.
    opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=5e-4,nesterov=True)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    best=-1; best_state=None; started=time.perf_counter()
    for ep in range(args.epochs):
        model.train(); loss_sum=0; correct=total=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=F.cross_entropy(logits,y,label_smoothing=.05)
            loss.backward(); opt.step()
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); total+=len(y)
        model.eval(); vc=vt=0
        with torch.no_grad():
            for x,y in eval_loaders[1]:
                x,y=x.to(device),y.to(device); vc+=(model(x).argmax(1)==y).sum().item(); vt+=len(y)
        vacc=100*vc/vt; sched.step()
        if vacc>best: best=vacc; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"Epoch {ep+1:02d}/{args.epochs}: train_loss={loss_sum/total:.4f} train_acc={100*correct/total:.2f}% val_acc={vacc:.2f}%"+(" *BEST*" if vacc==best else ""))
    model.load_state_dict(best_state); model.eval()
    train_time=time.perf_counter()-started
    # Extract identical frozen penultimate features for all five heads.
    backbone=ResNetFeatures(pretrained=False).to(device)
    backbone.net.load_state_dict({k:v for k,v in model.state_dict().items() if not k.startswith("fc.")},strict=True)
    backbone.eval()
    feats=[]; labels=[]; extraction=time.perf_counter()
    with torch.no_grad():
        for loader in eval_loaders:
            h=[]; ylist=[]
            for x,y in loader:
                h.append(backbone(x.to(device)).cpu()); ylist.append(y.cpu())
            feats.append(torch.cat(h)); labels.append(torch.cat(ylist))
    Htr,Hv,Hte=feats; ytr,yv,yte=labels
    # Standardize from training statistics only. This is essential for optimizer stability.
    mu=Htr.mean(0,keepdim=True); sd=Htr.std(0,keepdim=True).clamp_min(1e-5)
    Htr=(Htr-mu)/sd; Hv=(Hv-mu)/sd; Hte=(Hte-mu)/sd
    ext_time=time.perf_counter()-extraction
    print(f"\nFrozen representation: {Htr.shape[1]} dimensions | backbone best val={best:.2f}%")
    rows=[]
    def report(name,accs,sec):
        rows.append((name,accs["train"],accs["validation"],accs["test"],sec))
    w,acc,t=fit_and_evaluate_ridge(Htr,ytr,Hv,yv,Hte,yte,lambda_relative=1e-4); report("Real LS",acc,t)
    (_,_,_),acc,t=evaluate_binarized_head(Htr,ytr,Hv,yv,Hte,yte,w); report("Binarized LS",acc,t)
    (_, _),acc,t=evaluate_direct_binary_head(Htr,ytr,Hv,yv,Hte,yte,steps=1500,lr=.01,seed=args.seed); report("Direct Binary",acc,t)
    _,acc,t=evaluate_ste_head(Htr,ytr,Hv,yv,Hte,yte,epochs=500,lr=.003,seed=args.seed); report("STE Binary",acc,t)
    if args.qubo_tune:
        winner,_=tune_from_representations(Htr,ytr,Hv,yv,args.seed,(25,50),(250,500),(.25,.5))
        if winner is None: raise RuntimeError("All QUBO validation candidates failed.")
        nr,ns,mult=winner["num_reads"],winner["num_sweeps"],winner["scale_multiplier"]
        qw,qs,qb=winner["_weights"],winner["_scales"],winner["_bias"]
        t=float(winner["runtime_sec"])
    else:
        nr,ns,mult=args.num_reads,args.num_sweeps,.5
        qw,qs,qb,t=fit_fixed_qubo(Htr,ytr,num_reads=nr,num_sweeps=ns,seed=args.seed)
    qa={"train":evaluate_qubo(Htr,ytr,qw,qs,qb),"validation":evaluate_qubo(Hv,yv,qw,qs,qb),"test":evaluate_qubo(Hte,yte,qw,qs,qb)}
    report("QUBO Binary",qa,t)
    print("\n"+"="*76); print(f"{'HEAD':22s} {'TRAIN':>9s} {'VAL':>9s} {'TEST':>9s} {'TIME(s)':>10s}")
    for r in rows: print(f"{r[0]:22s} {r[1]:8.2f}% {r[2]:8.2f}% {r[3]:8.2f}% {r[4]:10.2f}")
    print("="*76)
    print(f"QUBO settings: reads={nr}, sweeps={ns}, multiplier={mult}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"seed":args.seed,"best_validation_accuracy":best,"epochs":args.epochs},args.output_dir/f"resnet18_cifar10_seed_{args.seed}.pt")
    import csv
    with open(args.output_dir/f"heads_seed_{args.seed}.csv","w",newline="") as f:
        wr=csv.writer(f); wr.writerow(["head","train_accuracy","validation_accuracy","test_accuracy","fit_seconds"]); wr.writerows(rows)
    print(f"Saved model and head results under {args.output_dir}")

if __name__=="__main__": main()
