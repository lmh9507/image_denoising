from __future__ import division
import os
import gc
import time
import glob
import datetime
import argparse
import json
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import optuna

from arch_unet import UNet, RESNET, ImprovedUNet
from util import L1FFT


# ─── Argument Parser ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--noisetype", type=str, default="gauss25")
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--save_model_path', type=str, default='./results')
parser.add_argument('--log_name', type=str, default='UNetImproved')
parser.add_argument('--gpu_devices', default='0', type=str,
                    help='Comma-separated list of GPU ids to use, e.g. "0,4,5,6" (ignored when HARD_VISIBLE_GPUS is set)')
parser.add_argument('--parallel', action='store_true')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--n_snapshot', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=4)
parser.add_argument('--n_trials', type=int, default=4,
                    help='Number of Optuna trials')
parser.add_argument('--tune_epochs', type=int, default=1,
                    help='HPO: fixed number of epochs per trial (objective uses last-epoch loss)')
args = parser.parse_args()

# ─── Hardcode CUDA_VISIBLE_DEVICES ──────────────────────────────────────────
# Hard-code the physical GPU IDs, e.g., [0,4,5,6]
HARD_VISIBLE_GPUS = [0, 4, 5, 6]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, HARD_VISIBLE_GPUS))
print(f"[INFO] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} | mapped GPUs: {list(range(torch.cuda.device_count()))}")
# Note: This applies only to this Python process and must be set before torch’s first CUDA call.

# ─── Determinism & Seeding ───────────────────────────────────────────────────
# Strong determinism settings
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
except Exception:
    pass
# Disable TF32 for extra consistency
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

SEED = 2025

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    except Exception:
        pass

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

set_seed(SEED)

# ─── Utils ──────────────────────────────────────────────────────────────────
def checkpoint(net, epoch, name):
    save_dir = os.path.join(args.save_model_path, args.log_name, systime)
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'epoch_{name}_{epoch:03d}.pth')
    torch.save(net.state_dict(), path)
    print(f'Checkpoint saved to {path}')

# ─── Dataset Definition (always [0,1] float32) ──────────────────────────────
class DenoiseDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()
        clean_paths = sorted(glob.glob(os.path.join(data_dir, 'clean', '*')))
        noisy_paths = sorted(glob.glob(os.path.join(data_dir, 'noise', '*')))
        self.pairs = list(zip(clean_paths, noisy_paths))
        # Safely convert 8/16-bit images to [0,1] float32
        self.transform = transforms.Compose([
            transforms.PILToTensor(),                       # uint8/uint16 tensor
            transforms.ConvertImageDtype(torch.float32),    # → float32, [0,1]
        ])
        print(f'Found {len(self.pairs)} samples')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fp_clean, fp_noisy = self.pairs[idx]
        clean = Image.open(fp_clean).convert('L')
        noisy = Image.open(fp_noisy).convert('L')
        return self.transform(clean), self.transform(noisy)

# ─── Training Helper ────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, criterion,
                    grad_clip=1.0, max_loss_skip=5.0, max_grad_norm=20.0):
    model.train()
    total_loss = 0.0
    for clean, noisy in loader:
        clean, noisy = clean.to(device), noisy.to(device)

        optimizer.zero_grad(set_to_none=True)
        out = model(noisy)

        # Validate outputs/targets
        if not torch.isfinite(out).all() or not torch.isfinite(clean).all():
            # print("[warn] non-finite tensor in batch → skip")
            continue

        loss = criterion(out, clean)

        # Skip batches with abnormally large loss (protect against data outliers)
        if not torch.isfinite(loss) or loss.item() > max_loss_skip:
            # print(f"[warn] abnormal loss={loss.item():.3e} → skip batch")
            continue

        loss.backward()

        # Secondary guard via gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if not np.isfinite(total_norm) or total_norm > max_grad_norm * 10:
            # print(f"[warn] abnormal grad norm={total_norm:.3e} → skip step")
            optimizer.zero_grad(set_to_none=True)
            continue

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(1, len(loader))


# ─── Setup: GPUs, Dataset, Split (global) ───────────────────────────────────
# Choose GPUs for tuning (after CUDA_VISIBLE_DEVICES remapping)
# GPU indices visible internally are now remapped to 0..N-1
PHYSICAL_GPUS = list(range(torch.cuda.device_count()))

systime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
SNAP_DIR = os.path.join(args.save_model_path, args.log_name, systime, "trial_init_snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

# Prepare Dataset ONCE (safe to share)
ds = DenoiseDataset(args.data_dir)
train_len = len(ds)
val_len = len(ds) - train_len
# Deterministic split
g_split = torch.Generator().manual_seed(SEED)
train_ds, _ = random_split(ds, [train_len, val_len], generator=g_split)

# Trial-local DataLoader factory (safe for n_jobs>1)
def make_train_loader(seed: int):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        train_ds,
        batch_size=args.batchsize,
        shuffle=True,
        generator=g,
        num_workers=0,          # keep CPU light for parallel trials
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

# ─── JSON Logger Callback ───────────────────────────────────────────────────
def json_logger(study, trial):
    rec = {
        'trial': trial.number,
        'value': trial.value,
        'params': trial.params,
        'completed': trial.datetime_complete.isoformat(),
        'init_seed': int(trial.user_attrs.get('init_seed', -1)),
        'device': int(trial.user_attrs.get('device', -1)),
    }
    with open('trials.log','a') as f:
        f.write(json.dumps(rec)+'\n')

# ─── Objective for HPO (fixed tune_epochs; last-epoch loss) ────────────────
def objective(trial):
    # select device for this trial
    if PHYSICAL_GPUS:
        idx = trial.number % len(PHYSICAL_GPUS)
        gid = PHYSICAL_GPUS[idx]
        device = torch.device(f'cuda:{gid}') if torch.cuda.is_available() else torch.device('cpu')
    else:
        gid = -1
        device = torch.device('cpu')

    # ensure this thread uses the intended GPU as default
    if device.type == 'cuda':
        torch.cuda.set_device(gid)

    # trial-unique seed → record only (do not mutate global RNG in multi-thread)
    init_seed = SEED + trial.number
    trial.set_user_attr("init_seed", int(init_seed))
    trial.set_user_attr("device", int(gid))

    # trial-local DataLoader (thread-safe)
    train_loader = make_train_loader(init_seed)

    # sample HP (epochs are fixed by CLI: args.tune_epochs)
    n_feature = trial.suggest_int('n_feature', 16, 128, step=16)
    lr = trial.suggest_loguniform('lr', 1e-6, 3e-4)
    tune_epochs = max(1, args.tune_epochs)

    # Fork RNG locally so other trials' RNG states are unaffected
    fork_devices = [gid] if gid >= 0 else []
    with torch.random.fork_rng(devices=fork_devices, enabled=True):
        torch.manual_seed(init_seed)
        if gid >= 0:
            torch.cuda.manual_seed_all(init_seed)

        # build model (single GPU per trial; avoid DataParallel here)
        model = ImprovedUNet(in_nc=1, out_nc=1, n_feature=n_feature).to(device)

        # save init weights snapshot for exact reproducibility later
        snap_path = os.path.join(SNAP_DIR, f"trial_{trial.number}_init.pth")
        state_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(state_cpu, snap_path)
        trial.set_user_attr("init_weight_path", snap_path)
        # lightweight checksum for sanity
        w0 = next(model.parameters()).detach()
        trial.set_user_attr("w0_sum", float(w0.sum().item()))

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        # milestones based on tune_epochs and strictly increasing (< tune_epochs)
        m1 = max(1, int(round(0.5 * tune_epochs)))
        m2 = max(m1 + 1, int(round(0.75 * tune_epochs)))
        milestones = [m for m in (m1, m2) if m < tune_epochs]
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.5,
        )
        criterion = nn.L1Loss()

        try:
            last_loss = None
            for e in range(tune_epochs):
                last_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
                scheduler.step()
            return float(last_loss)  # use last-epoch loss as objective
        finally:
            # cleanup VRAM for this device
            del model, optimizer, scheduler, criterion, train_loader
            gc.collect()
            if gid >= 0:
                torch.cuda.synchronize(device)
                with torch.cuda.device(gid):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

# ─── Main: Hyperparameter Search (multi-GPU) then Final Training ────────────
if __name__ == '__main__':
    # Hyperparameter search (true parallel across GPUs)
    storage = 'sqlite:///optuna_unet.db'
    study = optuna.create_study(
        study_name='unet_opt', storage=storage,
        load_if_exists=True, direction='minimize'
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=len(PHYSICAL_GPUS) if PHYSICAL_GPUS else 1,
        callbacks=[json_logger]
    )

    # best params
    best = study.best_trial.params
    print('Best hyperparams:', best)

    # extra cleanup after tuning
    for gid in PHYSICAL_GPUS:
        with torch.cuda.device(gid):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # restore exact seed/device conditions from the best trial
    best_seed = study.best_trial.user_attrs.get("init_seed", SEED)
    best_dev  = study.best_trial.user_attrs.get("device", (PHYSICAL_GPUS[0] if PHYSICAL_GPUS else -1))
    set_seed(best_seed)  # safe: final training is single-threaded

    # use the SAME GPU as the best trial for strict parity
    device = torch.device(f'cuda:{best_dev}') if best_dev >= 0 else torch.device('cpu')

    # build model BEFORE creating DataLoader to minimize any RNG drift
    net = ImprovedUNet(in_nc=1, out_nc=1, n_feature=best['n_feature']).to(device)

    # load exact init weights from the best trial, if available
    init_w_path = study.best_trial.user_attrs.get("init_weight_path", None)
    if init_w_path and os.path.isfile(init_w_path):
        sd = torch.load(init_w_path, map_location=device)
        net.load_state_dict(sd, strict=True)
        w0 = next(net.parameters()).detach()
        print(f"[repro] loaded init snapshot; w0_sum={w0.sum().item():.6f} vs trial {study.best_trial.user_attrs.get('w0_sum')}")
    else:
        print("[repro] init snapshot not found; using seeded init.")

    # trial-parity train loader (same shuffle seed)
    final_train_loader = make_train_loader(best_seed)

    # keep DataParallel OFF for strict reproducibility
    optimizer = optim.Adam(net.parameters(), lr=best['lr'], weight_decay=1e-5)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(0.5*args.n_epoch), int(0.75*args.n_epoch)],
        gamma=0.5,
    )
    criterion = nn.L1Loss()

    # logging paths
    systime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    save_dir = os.path.join(args.save_model_path, args.log_name, systime)
    validation_path = os.path.join(save_dir, "validation")
    os.makedirs(validation_path, exist_ok=True)
    log_path = os.path.join(validation_path, "A_log.csv")
    with open(log_path, "a") as f:
        f.write(f"epoch, loss, train_time\n")

    # Inference transform identical to training
    infer_transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ])

    for epoch in range(1, args.n_epoch+1):
        start = time.time()
        loss = train_one_epoch(net, final_train_loader, optimizer, device, criterion)
        scheduler.step()
        train_time = time.time()-start
        print(f'Epoch {epoch}: loss={loss:.4f}, time={train_time:.1f}s')

        if epoch % args.n_snapshot==0 or epoch==args.n_epoch:
            checkpoint(net, epoch, 'final')

        # log to CSV
        with open(log_path, "a") as f:
            f.write(f"{epoch}, {loss}, {train_time}\n")

        # every 10 epochs: save first image inference result
        if epoch % 10 == 0:
            first_idx = train_ds.indices[0] if hasattr(train_ds, 'indices') and len(train_ds.indices) > 0 else 0
            clean_fp, noisy_fp = ds.pairs[first_idx]

            origin = Image.open(clean_fp)
            noisy_img = Image.open(noisy_fp)

            noisy_t = infer_transform(noisy_img).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = net(noisy_t)
            pred = pred.permute(0, 2, 3, 1).cpu().data.clamp(0, 1).numpy().squeeze()
            pred255 = np.clip(pred * 255.0 + 0.5, 0, 255).astype(np.uint8)

            clean_name = os.path.splitext(os.path.basename(clean_fp))[0]
            noise_name = os.path.splitext(os.path.basename(noisy_fp))[0]

            if epoch == 10:
                Image.fromarray(np.array(origin)).convert('RGB').save(
                    os.path.join(validation_path, f"{clean_name}_000-{epoch:03d}_clean.png"))
                Image.fromarray(np.array(noisy_img)).convert('RGB').save(
                    os.path.join(validation_path, f"{noise_name}_000-{epoch:03d}_noisy.png"))
            Image.fromarray(pred255).convert('RGB').save(
                os.path.join(validation_path, f"{noise_name}_000-{epoch:03d}_denoised.png"))
