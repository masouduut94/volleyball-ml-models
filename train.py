"""
Unified VideoMAE (v1 or v2) fine-tuning for volleyball event classification.

Dataset layout:
    DATA_ROOT/
        active_play/*.mp4
        no_play/*.mp4
        breaks/*.mp4
        service/*.mp4

Install:
    pip install torch torchvision transformers timm opencv-python numpy

Usage: edit the settings in the `if __name__ == "__main__":` block at the bottom and run
    python train_videomae_unified.py
"""
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T
from transformers import AutoConfig, AutoModel, VideoMAEForVideoClassification

CLASSES = ["active_play", "no_play", "breaks", "service"]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODELS = {1: "MCG-NJ/videomae-base", 2: "OpenGVLab/VideoMAEv2-Base"}


# =========================================================================== #
# Data
# =========================================================================== #
def list_videos(root):
    items = []
    for label, cls in enumerate(CLASSES):
        files = sorted(p for p in (Path(root) / cls).rglob("*") if p.suffix.lower() in VIDEO_EXTS)
        items += [(str(p), label) for p in files]
        print(f"{cls:12s}: {len(files)} videos")
    return items


def stratified_split(items, val_ratio, seed):
    rng = random.Random(seed)
    train, val = [], []
    for label in range(len(CLASSES)):
        cls_items = [it for it in items if it[1] == label]
        rng.shuffle(cls_items)
        n_val = max(1, int(len(cls_items) * val_ratio))
        val += cls_items[:n_val]
        train += cls_items[n_val:]
    return train, val


def read_all_frames(path):
    """Read every frame as RGB uint8. Clips are <= 3 s so this is cheap."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def sample_indices(n_avail, num_frames, train):
    """Split the window into `num_frames` segments; random frame per segment (train)
    or the segment centre (eval). Repeats frames if the window is shorter than num_frames."""
    seg = n_avail / num_frames
    if train:
        idx = [int(seg * i + random.random() * seg) for i in range(num_frames)]
    else:
        idx = [int(seg * i + seg / 2) for i in range(num_frames)]
    return [min(max(i, 0), n_avail - 1) for i in idx]


class VolleyballClips(Dataset):
    """Returns (clip, label) with clip shaped (T, C, H, W), normalized."""

    def __init__(self, items, image_size=224, num_frames=16, train=True,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.items, self.num_frames, self.train = items, num_frames, train
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        short = int(image_size * 256 / 224)
        if train:  # torchvision v2 applies identical random params to every frame in (T,C,H,W)
            self.tf = T.Compose([T.Resize(short, antialias=True), T.RandomCrop(image_size),
                                 T.RandomHorizontalFlip(0.5)])
        else:
            self.tf = T.Compose([T.Resize(short, antialias=True), T.CenterCrop(image_size)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        frames, fps = read_all_frames(path)
        if not frames:
            raise RuntimeError(f"Could not read any frame from {path}")

        win = int(round(fps))                                  # ~30 frames = 1 second
        max_start = max(0, len(frames) - win)
        start = random.randint(0, max_start) if self.train else 0
        window = frames[start:start + win]

        idx = sample_indices(len(window), self.num_frames, self.train)   # 16 of ~30
        clip = np.stack([window[j] for j in idx])
        clip = torch.from_numpy(clip).permute(0, 3, 1, 2)                # (T,3,H,W)
        clip = self.tf(clip).float() / 255.0
        clip = (clip - self.mean) / self.std
        return clip, label


# =========================================================================== #
# Unified model
# =========================================================================== #
class VideoMAEClassifier(nn.Module):
    """One interface for both VideoMAE versions.

    forward(x): x is (B, T, C, H, W) -> logits (B, num_classes)

    version=1: Hugging Face native VideoMAEForVideoClassification.
    version=2: OpenGVLab VideoMAE V2 backbone (remote code) + linear head.
    """

    def __init__(self, version, model_name=None, num_classes=len(CLASSES), dropout=0.1,
                 num_frames=16, image_size=224):
        super().__init__()
        assert version in (1, 2), "version must be 1 or 2"
        self.version = version
        self.model_name = model_name or DEFAULT_MODELS[version]
        self.num_frames, self.image_size = num_frames, image_size

        if version == 1:
            self.model = VideoMAEForVideoClassification.from_pretrained(
                self.model_name,
                num_labels=num_classes,
                id2label=dict(enumerate(CLASSES)),
                label2id={c: i for i, c in enumerate(CLASSES)},
                ignore_mismatched_sizes=True,
            )
            self.num_frames = self.model.config.num_frames
            self.image_size = self.model.config.image_size
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.backbone = AutoModel.from_pretrained(self.model_name, config=config,
                                                      trust_remote_code=True)
            if hasattr(self.backbone, "head"):        # we want pooled features, not its logits
                self.backbone.head = nn.Identity()
            self.backbone.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, self.num_frames, image_size, image_size)
                feat_dim = self.backbone(dummy).shape[-1]
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(feat_dim, num_classes))

    def forward(self, x):
        if self.version == 1:
            return self.model(pixel_values=x).logits
        return self.head(self.backbone(x.permute(0, 2, 1, 3, 4)))   # V2 wants (B,C,T,H,W)

    def param_groups(self, lr, head_lr):
        """Backbone gets `lr`, the freshly initialised head gets `head_lr`."""
        if self.version == 1:
            head = list(self.model.classifier.parameters())
            fc_norm = getattr(self.model, "fc_norm", None)
            if fc_norm is not None:
                head += list(fc_norm.parameters())
        else:
            head = list(self.head.parameters())
        head_ids = {id(p) for p in head}
        body = [p for p in self.parameters() if id(p) not in head_ids]
        return [{"params": body, "lr": lr}, {"params": head, "lr": head_lr}]

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "version": self.version,
                    "model_name": self.model_name, "classes": CLASSES}, path)


# =========================================================================== #
# Trainer
# =========================================================================== #
class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = VideoMAEClassifier(
            cfg["version"], cfg["model_name"], len(CLASSES), cfg["dropout"],
            cfg["num_frames"], cfg["image_size"]).to(self.device)

        items = list_videos(cfg["data_root"])
        train_items, val_items = stratified_split(items, cfg["val_ratio"], cfg["seed"])
        print(f"train={len(train_items)}  val={len(val_items)}")

        kw = dict(image_size=self.model.image_size, num_frames=self.model.num_frames)
        self.train_dl = DataLoader(VolleyballClips(train_items, train=True, **kw), cfg["batch_size"],
                                   shuffle=True, num_workers=cfg["num_workers"],
                                   pin_memory=True, drop_last=True)
        self.val_dl = DataLoader(VolleyballClips(val_items, train=False, **kw), cfg["batch_size"],
                                 shuffle=False, num_workers=cfg["num_workers"])

        counts = np.bincount([l for _, l in train_items], minlength=len(CLASSES)).clip(min=1)
        w = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float, device=self.device)
        self.criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg["label_smoothing"])

        self.opt = torch.optim.AdamW(self.model.param_groups(cfg["lr"], cfg["head_lr"]),
                                     weight_decay=cfg["weight_decay"])
        total = cfg["epochs"] * len(self.train_dl)
        warmup = max(1, int(0.1 * total))
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: (s + 1) / warmup if s < warmup
            else 0.5 * (1 + np.cos(np.pi * (s - warmup) / max(1, total - warmup))))
        self.scaler = torch.amp.GradScaler(enabled=self.device == "cuda")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct = total = 0
        conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long)
        for x, y in self.val_dl:
            x, y = x.to(self.device), y.to(self.device)
            preds = self.model(x).argmax(-1)
            correct += (preds == y).sum().item()
            total += y.numel()
            for t, p in zip(y.cpu(), preds.cpu()):
                conf[t, p] += 1
        return correct / max(total, 1), conf

    def train(self):
        amp = self.device == "cuda"
        best = 0.0
        for epoch in range(1, self.cfg["epochs"] + 1):
            self.model.train()
            run_loss, n = 0.0, 0
            for x, y in self.train_dl:
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=amp):
                    loss = self.criterion(self.model(x), y)
                self.opt.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.opt); self.scaler.update(); self.sched.step()
                run_loss += loss.item() * y.size(0); n += y.size(0)

            acc, conf = self.evaluate()
            print(f"epoch {epoch:02d} | loss {run_loss / n:.4f} | val acc {acc:.4f}")
            if acc > best:
                best = acc
                self.model.save(self.cfg["output_path"])
                print(f"  saved best ({best:.4f}) -> {self.cfg['output_path']}")
                print(f"  confusion (rows=true, cols=pred), order={CLASSES}\n{conf}")
        return best


# =========================================================================== #
# Run
# =========================================================================== #
if __name__ == "__main__":
    # ---- settings: edit these ------------------------------------------------
    DATA_ROOT = "/home/masoud/Desktop/volleyball_datasets/Gamestate/combinations"
    # contains active_play/, no_play/, breaks/, service/
    VERSION = 2                                 # 1 -> HF VideoMAE v1, 2 -> VideoMAE V2 (remote code)
    MODEL_NAME = "OpenGVLab/VideoMAEv2-Base"         # None = default for VERSION, or e.g.
                                                # "MCG-NJ/videomae-small-finetuned-kinetics"
    NUM_FRAMES = 16                             # frames fed to the model (sampled from ~30)
    IMAGE_SIZE = 224
    EPOCHS = 15
    BATCH_SIZE = 8
    LR = 5e-5                                   # backbone learning rate
    HEAD_LR = 1e-3                              # learning rate for the new classification head
    WEIGHT_DECAY = 0.05
    DROPOUT = 0.1                               # used by the V2 head
    LABEL_SMOOTHING = 0.1
    VAL_RATIO = 0.2
    NUM_WORKERS = 4
    SEED = 42
    OUTPUT_PATH = f"videomae_v{VERSION}_volleyball.pt"
    # --------------------------------------------------------------------------

    cfg = dict(
        data_root=DATA_ROOT, version=VERSION, model_name=MODEL_NAME,
        num_frames=NUM_FRAMES, image_size=IMAGE_SIZE, epochs=EPOCHS,
        batch_size=BATCH_SIZE, lr=LR, head_lr=HEAD_LR, weight_decay=WEIGHT_DECAY,
        dropout=DROPOUT, label_smoothing=LABEL_SMOOTHING, val_ratio=VAL_RATIO,
        num_workers=NUM_WORKERS, seed=SEED, output_path=OUTPUT_PATH,
    )
    best_acc = Trainer(cfg).train()
    print(f"Best val acc: {best_acc:.4f}")