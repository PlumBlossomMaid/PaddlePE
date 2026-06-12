"""Test: PaddlePE Client 模式

验证场景：PyTorch 先导入，Paddle 因 cuDNN DLL 冲突无法加载（或 force_remote）。
测试内容：
  1. PE.list_models() -> 返回模型列表（通过 server）
  2. PE.create("fcpe", force_remote=True) -> RemotePE
  3. infer() 接受 numpy / torch.Tensor
  4. infer() 拒绝非法输入 -> TypeError
  5. training/export 不可导入 -> RuntimeError
  6. registry 不可用
  7. forward() -> NotImplementedError
  8. register() -> NotImplementedError
"""

import subprocess
import sys

TEST_CODE = r"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("[paddlePE Remote/Client Mode Test]")
print("=" * 60)

# -----------------------------------------------------------------------
# 0. 导入 PyTorch 并尝试 Paddle
# -----------------------------------------------------------------------
import torch
print(f"  [OK] PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})")

if torch.cuda.is_available():
    x = torch.tensor([1.0], device='cuda')
    print(f"  [OK] PyTorch CUDA tensor: {x}")

# 检测是否有 DLL 冲突
_HAS_DLL_CONFLICT = False
try:
    import paddle
    print(f"  [INFO] No DLL conflict - Paddle also available")
except Exception as e:
    _HAS_DLL_CONFLICT = True
    print(f"  [OK] DLL conflict detected: {e}")

# -----------------------------------------------------------------------
# 1. 获取 RemotePE 实例
# -----------------------------------------------------------------------
import numpy as np
import time

if _HAS_DLL_CONFLICT:
    # 真正 Client 模式：自动回退
    from paddlepe import PE
    print(f"  [OK] from paddlepe import PE (auto client mode)")

    t0 = time.time()
    models = PE.list_models()
    print(f"  [OK] PE.list_models() = {models} ({time.time()-t0:.1f}s)")

    pe = PE.create("fcpe")
    print(f"  [OK] PE.create('fcpe') -> {type(pe).__name__}")
else:
    # 无冲突环境：force_remote 模拟
    from paddlepe import PE
    print(f"  [OK] from paddlepe import PE")

    t0 = time.time()
    models = PE.list_models()
    print(f"  [OK] PE.list_models() = {models} ({time.time()-t0:.1f}s)")

    pe = PE.create("fcpe", force_remote=True)
    print(f"  [OK] PE.create('fcpe', force_remote=True) -> {type(pe).__name__}")

# -----------------------------------------------------------------------
# 2. infer() - 各种输入类型
# -----------------------------------------------------------------------
sr = 16000
t_len = int(sr * 0.3)
wav_np = np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, t_len)).astype(np.float32)

# 2a. numpy
t0 = time.time()
f0, conf = pe.infer(wav_np, sr)
assert f0 is not None and f0.shape[0] > 0
print(f"  [OK] pe.infer(numpy) -> f0:{f0.shape} ({time.time()-t0:.1f}s)")

# 2b. torch.Tensor（自动 .numpy()）
wav_th = torch.from_numpy(wav_np)
f0, conf = pe.infer(wav_th, sr)
assert f0 is not None
print(f"  [OK] pe.infer(torch.Tensor) -> f0:{f0.shape}")

# 2c. 非法输入 -> TypeError
try:
    pe.infer("not a tensor", sr)
    print("  [FAIL] should raise TypeError")
    sys.exit(1)
except TypeError:
    print(f"  [OK] pe.infer(str) -> TypeError")

# 2d. get_pitch 别名
f0, conf = pe.get_pitch(wav_np, sr)
assert f0 is not None
print(f"  [OK] pe.get_pitch(numpy) -> f0:{f0.shape}")

# -----------------------------------------------------------------------
# 3. forward() -> NotImplementedError / training/export guard
# -----------------------------------------------------------------------
try:
    pe.forward(wav_np)
    print("  [FAIL] forward() should raise")
    sys.exit(1)
except NotImplementedError:
    print(f"  [OK] pe.forward() -> NotImplementedError")

try:
    from paddlepe.training import PETask
    print("  [FAIL] training should not be importable")
    sys.exit(1)
except RuntimeError:
    print(f"  [OK] training import -> RuntimeError")

try:
    from paddlepe.export import export_onnx
    print("  [FAIL] export should not be importable")
    sys.exit(1)
except RuntimeError:
    print(f"  [OK] export import -> RuntimeError")

if _HAS_DLL_CONFLICT:
    try:
        from paddlepe import registry
        print("  [FAIL] registry should not be available")
        sys.exit(1)
    except ImportError:
        print(f"  [OK] registry not exported in client mode")

    try:
        PE.register("test_model")
        print("  [FAIL] register() should raise")
        sys.exit(1)
    except NotImplementedError:
        print(f"  [OK] PE.register() -> NotImplementedError")

# -----------------------------------------------------------------------
print("=" * 60)
print("[PASS] Remote/Client mode: ALL TESTS PASSED")
print("=" * 60)
"""

print("Starting subprocess for client/remote mode test...\n")

result = subprocess.run(
    [sys.executable, "-c", TEST_CODE],
    capture_output=True, text=True, timeout=300,
)

print(result.stdout)
if result.stderr:
    for line in result.stderr.splitlines():
        print(f"  [stderr] {line}")

if result.returncode != 0:
    sys.exit(1)

print("[PASS] All client mode tests passed")
