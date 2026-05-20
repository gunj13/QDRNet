# train.py
#
# Usage:
#   python train.py                 # Hadamard (default)
#   python train.py --ablation      # classical linear (no quantum)
#   python train.py --mode vqc      # VQC 
#   python train.py --resume        # continue from last checkpoint

import os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

import config as C
from data.dataset     import get_loaders
from model.classifier import DRNet
from utils            import compute_metrics

BEST_CKPT = os.path.join(C.CHECKPOINT_DIR, "best_model.pt")
LAST_CKPT = os.path.join(C.CHECKPOINT_DIR, "last_checkpoint.pt")
LOG_PATH  = os.path.join(C.CHECKPOINT_DIR, "training_log.pt")

# Class weights: DR0 is baseline 1.0; DR3/DR4 are rare and severely underweighted
# _CLASS_WEIGHTS = torch.tensor([1.0, 2.0, 2.5, 5.0, 5.0]) # trial
_CLASS_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])


# ── Loss ──────────────────────────────────────────────────────────────────

class FocalOrdinalLoss(nn.Module):
    """
    Focal loss + class weights + ordinal MSE penalty.

    Focal: down-weights easy DR0 examples, focuses on hard DR3/DR4.
    Ordinal: expected predicted grade penalised by distance from true grade.
             Predicting DR3 as DR0 (dist=3) costs more than predicting as DR2 (dist=1).
    """
    def __init__(self):
        super().__init__()
        self.gamma           = C.FOCAL_GAMMA
        self.label_smoothing = C.LABEL_SMOOTHING
        self.ordinal_weight  = C.ORDINAL_WEIGHT
        self.num_classes     = C.NUM_CLASSES

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        B, device = logits.size(0), logits.device

        cw = _CLASS_WEIGHTS.to(device)
        gi = torch.arange(self.num_classes, dtype=torch.float32, device=device)

        # Label smoothing
        with torch.no_grad():
            smooth = torch.full((B, self.num_classes),
                                self.label_smoothing / self.num_classes,
                                device=device)
            smooth.scatter_(1, targets.unsqueeze(1),
                            1.0 - self.label_smoothing
                            + self.label_smoothing / self.num_classes)

        log_p = F.log_softmax(logits, dim=1)     
        p     = log_p.exp()

        # Focal weighting
        p_t   = (p * smooth).sum(1)
        focal = (1 - p_t) ** self.gamma

        # Class-weighted cross-entropy
        ce         = -(smooth * log_p).sum(1)
        focal_loss = (focal * ce * cw[targets]).mean()

        # Ordinal MSE
        if self.ordinal_weight > 0:
            expected = (p * gi).sum(1)
            ordinal  = F.mse_loss(expected, targets.float())
            return focal_loss + self.ordinal_weight * ordinal       
            
        return focal_loss


# ── Backbone freeze helpers ────────────────────────────────────────────────

def freeze_backbone(model):
    for p in model.backbone.features.parameters():
        p.requires_grad = False

def unfreeze_backbone(model):
    for p in model.backbone.features.parameters():
        p.requires_grad = True


# ── Scheduler ─────────────────────────────────────────────────────────────

def make_scheduler(optimizer, total_epochs, warmup_epochs):
    return SequentialLR(
        optimizer,
        schedulers=[
            LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs),
            CosineAnnealingLR(optimizer,
                              T_max=total_epochs - warmup_epochs, eta_min=1e-6),
        ],
        milestones=[warmup_epochs])


# ── Epoch ─────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, preds, labels = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for fundus, masks, lbl in tqdm(loader, leave=False,
                                        desc="train" if is_train else "val"):
            fundus = fundus.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            lbl    = lbl.to(device, non_blocking=True)

            logits = model(fundus, masks)
            loss   = criterion(logits, lbl)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds  += logits.argmax(1).cpu().tolist()
            labels += lbl.cpu().tolist()

    return total_loss / len(loader), compute_metrics(labels, preds)


# ── Main ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hadamard", "vqc"], default=None,
                   help="Override interaction mode (default: config.INTERACTION_MODE)")
    p.add_argument("--ablation", action="store_true",
                   help="Replace quantum layer with plain linear")
    p.add_argument("--resume",   action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(C.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(C.RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nLoading data...")
    train_loader, val_loader, _ = get_loaders()

    print("\nBuilding model...")
    interaction_mode = args.mode if args.mode is not None else C.INTERACTION_MODE
    if args.ablation:
        model = DRNet(use_quantum=False)
    else:
        print(f"  Interaction mode: {interaction_mode}"
              f" ({'CLI override' if args.mode is not None else 'from config'})")
        model = DRNet(interaction_mode=interaction_mode)
    model = model.to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_p:,}")

    # Freeze backbone for first 10 epochs so new layers initialise before
    # the pretrained backbone starts adjusting to APTOS
    FREEZE_EPOCHS = 10
    freeze_backbone(model)
    print(f"  Backbone frozen for first {FREEZE_EPOCHS} epochs")
    backbone_unfrozen = False

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    scheduler  = make_scheduler(optimizer, C.EPOCHS, C.WARMUP_EPOCHS)
    criterion  = FocalOrdinalLoss()

    print(f"  Loss: FocalOrdinal | patience={C.PATIENCE} | "
          f"epochs={C.EPOCHS} | batch={C.BATCH_SIZE} | img={C.IMG_SIZE}px")

    start_epoch, best_f1, patience_ctr = 1, 0.0, 0
    log = {"epochs": [], "train_loss": [], "val_loss": [],
           "train_f1": [], "val_f1": [], "best_epoch": 0}

    if args.resume and os.path.exists(LAST_CKPT):
        ckpt = torch.load(LAST_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_f1      = ckpt["best_f1"]
        patience_ctr = ckpt.get("patience_ctr", 0)
        backbone_unfrozen = ckpt.get("backbone_unfrozen", False)
        if backbone_unfrozen:
            unfreeze_backbone(model)
        print(f"  Resumed from epoch {start_epoch-1}, best F1={best_f1:.2f}%")

    print(f"\nTraining epochs {start_epoch}–{C.EPOCHS}\n")

    for epoch in range(start_epoch, C.EPOCHS + 1):
        t0 = time.time()

        # Unfreeze backbone after FREEZE_EPOCHS at 10x lower lr
        if not backbone_unfrozen and epoch > FREEZE_EPOCHS:
            unfreeze_backbone(model)
            backbone_unfrozen = True
            optimizer = AdamW([
                {"params": model.backbone.features.parameters(),
                 "lr": C.LEARNING_RATE * 0.1},
                {"params": [p for n, p in model.named_parameters()
                            if "backbone.features" not in n],
                 "lr": C.LEARNING_RATE},
            ], weight_decay=C.WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=C.EPOCHS - epoch, eta_min=1e-6)
            print(f"  Ep {epoch}: backbone unfrozen at lr={C.LEARNING_RATE*0.1:.1e}")

        tr_loss, tr_m = run_epoch(
            model, train_loader, optimizer, criterion, device, True)
        vl_loss, vl_m = run_epoch(
            model, val_loader,   optimizer, criterion, device, False)
        scheduler.step()

        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Ep {epoch:3d}/{C.EPOCHS}  "
              f"loss {tr_loss:.3f}/{vl_loss:.3f}  "
              f"F1 {tr_m['f1']:.2f}/{vl_m['f1']:.2f}%  "
              f"ACC {vl_m['acc']:.2f}%  "
              f"lr {lr:.2e}  ({elapsed:.0f}s)")

        log["epochs"].append(epoch)
        log["train_loss"].append(tr_loss)
        log["val_loss"].append(vl_loss)
        log["train_f1"].append(tr_m["f1"])
        log["val_f1"].append(vl_m["f1"])

        if vl_m["f1"] > best_f1:
            best_f1, patience_ctr = vl_m["f1"], 0
            log["best_epoch"] = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_metrics": vl_m,
                "mode":        interaction_mode if not args.ablation else "ablation",
            }, BEST_CKPT)
            print(f"  ✓ New best — val F1 = {best_f1:.2f}%")
        else:
            patience_ctr += 1

        torch.save({
            "epoch":              epoch,
            "model_state":        model.state_dict(),
            "optimizer_state":    optimizer.state_dict(),
            "scheduler_state":    scheduler.state_dict(),
            "best_f1":            best_f1,
            "patience_ctr":       patience_ctr,
            "backbone_unfrozen":  backbone_unfrozen,
        }, LAST_CKPT)
        torch.save(log, LOG_PATH)

        if patience_ctr >= C.PATIENCE:
            print(f"\nEarly stopping: no val improvement for {C.PATIENCE} epochs.")
            break

    print(f"\nDone. Best val F1 = {best_f1:.2f}% at epoch {log['best_epoch']}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
