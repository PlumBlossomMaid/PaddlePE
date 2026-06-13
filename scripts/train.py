#!/usr/bin/env python
"""端到端训练：下载 MIR-1K → 统一 HDF5 → 任意模型训练。

用法：
    python scripts/train.py --model fcpe      # FCPE 训练
    python scripts/train.py --model rmvpe     # RMVPE 训练
    python scripts/train.py --model crepe     # CREPE 训练
    python scripts/train.py --model penn      # PENN 训练
"""

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _PROJECT_ROOT)
_OCEAN_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(_PROJECT_ROOT)), "paddleOcean"
)
sys.path.insert(0, _OCEAN_ROOT)

import ocean
import paddle

from paddlepe.training import PEDataModule, collators
from paddlepe.training.preprocess.preprocess_mir1k import (
    preprocess as preprocess_mir1k,
)

DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "mir1k")
H5_PATH = os.path.join(_PROJECT_ROOT, "data", "h5", "mir1k.h5")

# ====================================================================
# 模型注册表
# ====================================================================


def _model_registry():
    """返回 {模型名: (model_class, collator_class, description)}"""
    from paddlepe.models.crepe.backbone import CrepeBackbone
    from paddlepe.models.fcpe.backbone import MelConformerF0
    from paddlepe.models.penn.backbone import PennBackbone
    from paddlepe.models.rmvpe.backbone import RMVPEUNet

    return {
        "fcpe": (MelConformerF0, collators.FCPECollator, "MelConformer"),
        "rmvpe": (RMVPEUNet, collators.RMVPECollator, "UNet+BiGRU"),
        "crepe": (CrepeBackbone, collators.CREPECollator, "Conv2D+Sigmoid"),
        "penn": (PennBackbone, collators.PENNCollator, "FCNF0++"),
    }


def _build_model(model_name: str):
    """创建 ocean.Model 子类实例。"""
    registry = _model_registry()
    backbone_cls, _, desc = registry[model_name]

    losses = {
        "fcpe": (
            "bce",
            lambda p, t: paddle.nn.functional.binary_cross_entropy(p, t),
        ),
        "rmvpe": (
            "bce",
            lambda p, t: paddle.nn.functional.binary_cross_entropy(p, t),
        ),
        "crepe": (
            "mse",
            lambda p, t: paddle.nn.functional.mse_loss(p.squeeze(-1), t),
        ),
        "penn": (
            "mse",
            lambda p, t: paddle.nn.functional.mse_loss(p.squeeze(-1), t),
        ),
    }
    loss_name, loss_fn = losses[model_name]

    class _Model(ocean.Model):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_cls()
            self.loss_name = loss_name

        def forward(self, x):
            return self.backbone(x)

        def to(self, device, *args, **kwargs):
            # Workaround: skip to() if already on target device
            try:
                for p in self.parameters():
                    if p.numel() > 0:
                        if str(p.place) == str(device):
                            return self
                        break
            except Exception:
                pass
            return super().to(device, *args, **kwargs)

        def training_step(self, batch, batch_idx):
            inp, target = batch
            pred = self(inp)
            loss = loss_fn(pred, target)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            inp, target = batch
            pred = self(inp)
            loss = loss_fn(pred, target)
            self.log("val_loss", loss, prog_bar=True)

        def configure_optimizers(self):
            return ocean.optimizer.AdamW(
                learning_rate=1e-3,
                parameters=self.parameters(),
            )

    return _Model(), desc


# ====================================================================
# Main
# ====================================================================


def main():
    parser = argparse.ArgumentParser(description="PaddlePE 端到端训练")
    parser.add_argument(
        "--model",
        type=str,
        default="rmvpe",
        choices=["fcpe", "rmvpe", "crepe", "penn"],
        help="模型名称",
    )
    parser.add_argument("--epochs", type=int, default=1, help="训练 epoch 数")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", type=str, default="gpu:0", help="训练设备")
    parser.add_argument(
        "--val-split", type=float, default=0.02, help="验证集比例"
    )
    parser.add_argument(
        "--log-dir", type=str, default=None, help="VisualDL 日志目录"
    )
    parser.add_argument(
        "--no-forward-check", action="store_true", help="跳过前向验证步骤"
    )
    args = parser.parse_args()

    # ── Step 1: 下载 + 预处理 ──
    print("=" * 60)
    print("  Step 1/3: 下载 & 预处理 MIR-1K → 统一 HDF5")
    print("=" * 60)
    os.makedirs(os.path.dirname(H5_PATH), exist_ok=True)
    preprocess_mir1k(DATA_ROOT, H5_PATH, overwrite=False)

    # ── Step 2: 构建模型 + 数据 ──
    print("\n" + "=" * 60)
    print(f"  Step 2/3: 构建 {args.model.upper()} + PEDataModule")
    print("=" * 60)

    paddle.set_device(args.device)
    print(f"  设备: {paddle.get_device()}")

    model, desc = _build_model(args.model)
    params = sum(p.numel().numpy() for p in model.parameters())
    print(f"  模型: {args.model.upper()} ({desc})")
    print(f"  参数量: {params / 1e6:.1f}M")

    collator_cls = _model_registry()[args.model][1]
    dm = PEDataModule(
        datasets={"mir1k": DATA_ROOT},
        collator=collator_cls(),
        batch_size=args.batch_size,
        val_split=args.val_split,
    )
    dm.prepare_data()
    dm.setup("fit")

    if not args.no_forward_check:
        collator = collator_cls()
        sample = dm._train_ds[0]
        inp, target = collator([sample])
        out = model(inp)
        print(
            f"  输入: {inp.shape}  →  输出: {out.shape}  →  目标: {target.shape}"
        )

    # ── Step 3: 训练 ──
    print("\n" + "=" * 60)
    print(f"  Step 3/3: ocean.Trainer 训练 {args.epochs} epoch(s)")
    print("=" * 60)

    # 配置日志器
    loggers = []
    if args.log_dir:
        loggers.append(
            ocean.VisualDLLogger(save_dir=args.log_dir, name=args.model)
        )
    loggers.append(
        ocean.CSVLogger(root_dir=args.log_dir or "./logs", name=args.model)
    )

    trainer = ocean.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=loggers,
        log_every_n_steps=10,
        enable_checkpointing=False,
        enable_progress_bar=True,
        verbose=True,
    )
    trainer.fit(model, datamodule=dm)

    print(f"\n✅ {args.model.upper()} 端到端训练完成！")


if __name__ == "__main__":
    main()
