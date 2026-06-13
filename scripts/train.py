#!/usr/bin/env python
"""端到端训练：N 数据集 → 统一 HDF5 → 任意模型训练。

用法：
    # 默认配置 (mir1k + ptdb, rmvpe, 1 epoch)
    python scripts/train.py

    # 指定配置文件和 CLI 覆盖
    python scripts/train.py --config configs/train_default.yaml
    python scripts/train.py --model fcpe --datasets mir1k
    python scripts/train.py --model crepe --datasets ptdb --training.epochs 5
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
_OCEAN_ROOT = _PROJECT_ROOT.parent / "paddleOcean"
sys.path.insert(0, str(_OCEAN_ROOT))

import ocean  # noqa: E402
import paddle  # noqa: E402
import yaml  # noqa: E402

from paddlepe.training import PEDataModule, collators  # noqa: E402

# ====================================================================
# 工具：递归合并字典（target ← source）
# ====================================================================


def _deep_merge(base: dict, overlay: dict) -> dict:
    """递归合并 overlay 到 base，返回新字典。"""
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_set(cfg: dict, keys: list[str], value) -> dict:
    """在嵌套字典中设置 path.to.key = value，返回新字典。"""
    if len(keys) == 1:
        cfg[keys[0]] = value
    else:
        if keys[0] not in cfg or not isinstance(cfg[keys[0]], dict):
            cfg[keys[0]] = {}
        _deep_set(cfg[keys[0]], keys[1:], value)
    return cfg


# ====================================================================
# 默认配置（内嵌，无需 YAML 文件也能跑）
# ====================================================================

DEFAULT_CONFIG = {
    "model": "rmvpe",
    "datasets": ["mir1k", "ptdb"],
    "training": {
        "epochs": 1,
        "batch_size": 2,
        "device": "gpu:0",
        "val_split": 0.02,
        "num_workers": 0,
    },
    "optimizer": {
        "lr": 0.001,
        "weight_decay": 0.0,
    },
    "logging": {
        "log_dir": None,
        "log_every_n_steps": 10,
    },
    "checkpoint": {
        "enable": False,
        "dirpath": None,
        "every_n_epochs": 1,
    },
}

# ── 数据集注册表 ──
DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "mir1k": (
        str(_PROJECT_ROOT / "data" / "mir1k"),
        str(_PROJECT_ROOT / "data" / "h5" / "mir1k.h5"),
    ),
    "ptdb": (
        str(_PROJECT_ROOT / "data" / "ptdb"),
        str(_PROJECT_ROOT / "data" / "h5" / "ptdb.h5"),
    ),
}

# ====================================================================
# 模型注册表
# ====================================================================


def _model_registry():
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


def _build_model(model_name: str, cfg: dict):
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
    opt_cfg = cfg.get("optimizer", {})

    class _Model(ocean.Model):
        def __init__(self):
            super().__init__()
            self.backbone = backbone_cls()
            self.loss_name = loss_name

        def forward(self, x):
            return self.backbone(x)

        def to(self, device, *args, **kwargs):
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
                learning_rate=opt_cfg.get("lr", 0.001),
                parameters=self.parameters(),
            )

    return _Model(), desc


# ====================================================================
# 数据预处理
# ====================================================================


def _preprocess_datasets(datasets: list[str]) -> dict[str, str]:
    """确保所有数据集已预处理，返回 {name: root}。"""
    result = {}
    for name in datasets:
        if name in DATASET_REGISTRY:
            root, h5_path = DATASET_REGISTRY[name]
            result[name] = root
            if os.path.exists(h5_path):
                print(f"  ✓ {name}: HDF5 已存在")
                continue
            print(f"  → {name}: 预处理中 ...")
            try:
                mod = __import__(
                    f"paddlepe.training.preprocess.preprocess_{name}",
                    fromlist=["preprocess"],
                )
                os.makedirs(os.path.dirname(h5_path), exist_ok=True)
                mod.preprocess(root, h5_path)
                print(f"  ✓ {name}: 预处理完成")
            except ImportError:
                print(f"  ⚠  {name}: 无预处理脚本，跳过")
        else:
            custom_root = str(_PROJECT_ROOT / "data" / name)
            if os.path.isdir(custom_root):
                result[name] = custom_root
            else:
                print(f"  ⚠  未知数据集 '{name}'，跳过")
    return result


# ====================================================================
# CLI
# ====================================================================


def _parse_cli() -> tuple[dict, dict]:
    """解析 CLI 参数，返回 (config_overrides, flags)。

    顶层键直接覆盖: --model rmvpe
    嵌套键用点号:   --training.epochs 5
    布尔开关:       --no-forward-check, --verbose
    """
    parser = argparse.ArgumentParser(description="PaddlePE 端到端训练")
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["fcpe", "rmvpe", "crepe", "penn"],
    )
    parser.add_argument(
        "--datasets", type=str, default=None, help="逗号分隔的数据集列表"
    )
    parser.add_argument("--training.epochs", type=int, default=None)
    parser.add_argument("--training.batch_size", type=int, default=None)
    parser.add_argument("--training.device", type=str, default=None)
    parser.add_argument("--training.val_split", type=float, default=None)
    parser.add_argument("--training.num_workers", type=int, default=None)
    parser.add_argument("--optimizer.lr", type=float, default=None)
    parser.add_argument("--optimizer.weight_decay", type=float, default=None)
    parser.add_argument("--logging.log_dir", type=str, default=None)
    parser.add_argument("--logging.log_every_n_steps", type=int, default=None)
    parser.add_argument("--checkpoint.enable", type=bool, default=None)
    parser.add_argument("--checkpoint.dirpath", type=str, default=None)
    parser.add_argument("--checkpoint.every_n_epochs", type=int, default=None)
    parser.add_argument("--no-forward-check", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)

    parsed, _ = parser.parse_known_args()

    # 收集非 None 的覆盖项 → 嵌套字典
    overrides: dict = {}
    for key, val in vars(parsed).items():
        if val is None or key in ("config", "no_forward_check", "verbose"):
            continue
        parts = key.split(".")
        _deep_set(overrides, parts, val)

    flags = {
        "no_forward_check": parsed.no_forward_check,
        "verbose": parsed.verbose,
        "config_path": parsed.config,
    }
    return overrides, flags


# ====================================================================
# Main
# ====================================================================


def main():
    cli_overrides, flags = _parse_cli()

    # ── 加载配置（优先级: CLI YAML > 默认YAML > 内嵌默认） ──
    cfg = dict(DEFAULT_CONFIG)

    # 1) 如果指定了 --config，从该文件加载
    if flags["config_path"]:
        with open(flags["config_path"], encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, file_cfg)

    # 2) 尝试加载默认配置文件（可能不存在）
    default_yaml = _PROJECT_ROOT / "configs" / "train_default.yaml"
    if default_yaml.exists() and not flags["config_path"]:
        with open(default_yaml, encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, file_cfg)

    # 3) CLI override 优先级最高
    cfg = _deep_merge(cfg, cli_overrides)

    # ── 提取参数 ──
    model_name = cfg["model"]
    raw_datasets = cfg["datasets"]
    if isinstance(raw_datasets, str):
        dataset_names = [d.strip() for d in raw_datasets.split(",")]
    else:
        dataset_names = list(raw_datasets)
    train_cfg = cfg["training"]

    print("=" * 60)
    print("  PaddlePE 端到端训练")
    print("=" * 60)
    print(f"  模型: {model_name}")
    print(f"  数据集: {dataset_names}")

    # ── Step 1: 预处理 ──
    print("\n" + "=" * 60)
    print("  Step 1/3: 数据集预处理")
    print("=" * 60)
    datasets = _preprocess_datasets(dataset_names)

    if not datasets:
        print("  ❌ 没有可用数据集，退出")
        sys.exit(1)

    # ── Step 2: 构建模型 + DataModule ──
    print("\n" + "=" * 60)
    print("  Step 2/3: 构建模型 + DataModule")
    print("=" * 60)

    paddle.set_device(train_cfg["device"])
    print(f"  设备: {paddle.get_device()}")

    model, desc = _build_model(model_name, cfg)
    n_params = sum(p.numel().numpy() for p in model.parameters())
    print(f"  模型: {model_name.upper()} ({desc})")
    print(f"  参数量: {n_params / 1e6:.1f}M")

    collator_cls = _model_registry()[model_name][1]
    dm = PEDataModule(
        datasets=datasets,
        collator=collator_cls(),
        batch_size=train_cfg["batch_size"],
        val_split=train_cfg["val_split"],
        num_workers=train_cfg.get("num_workers", 0),
    )
    dm.prepare_data()
    dm.setup("fit")

    if not flags["no_forward_check"]:
        collator = collator_cls()
        sample = dm._train_ds[0]
        inp, target = collator([sample])
        out = model(inp)
        print(f"  输入: {inp.shape}  →  输出: {out.shape}  →  目标: {target.shape}")

    n_train = len(dm._train_ds)
    n_val = len(dm._val_ds) if dm._val_ds else 0
    print(f"  训练: {n_train} 样本, 验证: {n_val} 样本")

    # ── Step 3: 训练 ──
    print("\n" + "=" * 60)
    print(f"  Step 3/3: ocean.Trainer ({train_cfg['epochs']} epoch(s))")
    print("=" * 60)

    loggers = []
    log_dir = cfg.get("logging", {}).get("log_dir")
    if log_dir:
        loggers.append(ocean.VisualDLLogger(save_dir=log_dir, name=model_name))
    loggers.append(ocean.CSVLogger(root_dir=log_dir or "./logs", name=model_name))

    ckpt_cfg = cfg.get("checkpoint", {})
    trainer = ocean.Trainer(
        max_epochs=train_cfg["epochs"],
        accelerator="gpu",
        devices=1,
        logger=loggers,
        log_every_n_steps=cfg.get("logging", {}).get("log_every_n_steps", 10),
        enable_checkpointing=ckpt_cfg.get("enable", False),
        enable_progress_bar=True,
        verbose=flags.get("verbose", False),
    )
    trainer.fit(model, datamodule=dm)

    print(f"\n✅ {model_name.upper()} 训练完成！")


if __name__ == "__main__":
    main()
