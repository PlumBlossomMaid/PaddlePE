"""Standalone client mode test.

Run this in a fresh Python process (no PaddlePaddle pre-loaded):
    python scripts/test_client_standalone.py
"""

import sys
import time

import numpy as np

from paddlepe.client import ClientPE

print("=" * 60)
print("Client Mode Test (no PaddlePaddle)")
print("=" * 60)

# 1. list_models
t0 = time.time()
models = ClientPE.list_models()
print(f"[OK] ClientPE.list_models() = {models} ({(time.time() - t0):.1f}s)")
assert "fcpe" in models

# 2. create
pe = ClientPE.create("fcpe")
print(f"[OK] ClientPE.create('fcpe') -> {type(pe).__name__}")

# 3. infer with numpy
wav = np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, 4800)).astype(np.float32)
f0, conf = pe.infer(wav, 16000)
print(f"[OK] pe.infer(numpy) -> f0={f0.shape}")
assert f0.shape[0] > 0

# 4. infer with torch.Tensor (simulate PyTorch coexistence)
try:
    import torch

    wav_t = torch.from_numpy(wav)
    f0, conf = pe.infer(wav_t, 16000)
    print(f"[OK] pe.infer(torch.Tensor) -> f0={f0.shape}")
except ImportError:
    print("[SKIP] torch not installed")

# 5. illegal input -> TypeError
try:
    pe.infer("bad", 16000)
    print("[FAIL] should raise TypeError")
    sys.exit(1)
except TypeError:
    print("[OK] pe.infer(str) -> TypeError")

# 6. get_pitch alias
f0, conf = pe.get_pitch(wav, 16000)
print(f"[OK] pe.get_pitch(numpy) -> f0={f0.shape}")

# 7. forward -> NotImplementedError
try:
    pe.forward(wav)
    print("[FAIL] forward() should raise")
    sys.exit(1)
except NotImplementedError:
    print("[OK] pe.forward() -> NotImplementedError")

# 8. register -> NotImplementedError
try:
    ClientPE.register("x")
    print("[FAIL] register() should raise")
    sys.exit(1)
except NotImplementedError:
    print("[OK] ClientPE.register() -> NotImplementedError")

# 9. training/export -> clean error
from paddlepe._compat import require_paddle

try:
    require_paddle("test")
    print("[FAIL] require_paddle should raise")
    sys.exit(1)
except RuntimeError as e:
    msg = str(e)
    assert "PaddlePaddle is not available" in msg
    print("[OK] require_paddle() -> RuntimeError (clear message)")

pe.__del__()
print()
print("=" * 60)
print("[PASS] All client mode tests passed!")
print("=" * 60)
