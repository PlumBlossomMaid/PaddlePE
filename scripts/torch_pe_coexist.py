"""测试 PyTorch + PaddlePE 共存：先 import torch，再用 PaddlePE 推理。

验证 dual-mode 架构：
  1. PyTorch 占 CUDA → PE.create() 自动回退到 remote (子进程) 模式
  2. 推理结果正确返回
"""

import os
import sys

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)  # 项目根目录
sys.path.insert(0, os.path.join(_ROOT, "ocean_project", "PaddlePE"))
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

# ── Step 1: 先导入 torch（模拟 PyTorch 用户场景）──
print("=" * 60)
print("  Step 1/3: 导入 torch")
print("=" * 60)
import torch

print(f"  torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _ = torch.zeros([1, 1]).cuda()  # 让 PyTorch 占 CUDA
    print(f"  PyTorch owns CUDA (device 0: {torch.cuda.get_device_name(0)})")

# ── Step 2: 导入 paddlepe ──
print("\n" + "=" * 60)
print("  Step 2/3: 导入 paddlepe")
print("=" * 60)
from paddlepe import PE

print(f"  可用模型: {PE.list_models()}")

# ── Step 3: 推理一条音频 ──
print("\n" + "=" * 60)
print("  Step 3/3: FCPE 推理测试")
print("=" * 60)
import numpy as np
import soundfile as sf

wav_path = os.path.join(_ROOT, "体验", "标准歌声.flac")
wav, sr = sf.read(wav_path, dtype="float32")
if wav.ndim > 1:
    wav = wav.mean(axis=-1)
print(f"  音频: {len(wav)} samples @ {sr}Hz, {len(wav) / sr:.1f}s")

# 创建模型 — 自动回退到 remote 模式（因 PyTorch 已占 CUDA）
print("  创建 FCPE ...")
pe = PE.create("fcpe")
print(f"  模型类型: {type(pe).__name__}")

# Remote 模式输入 numpy，不要 import paddle
f0, conf = pe.infer(wav, sr)

f0_np = f0.numpy() if hasattr(f0, 'numpy') else np.array(f0)
valid = f0_np > 0
print(f"  结果: {len(f0_np)} 帧, 有声 {valid.sum()}")
if valid.any():
    print(f"  频率: [{f0_np[valid].min():.0f}, {f0_np[valid].max():.0f}] Hz")
    print(f"  中位数: {np.median(f0_np[valid]):.0f} Hz")

print("\n✅ PyTorch + PaddlePE 共存推理成功！")
