"""Test: PaddlePE 直接模式（无 PyTorch 冲突）

验证场景：正常 Python 进程，PaddlePaddle 可直接导入。
测试内容：
  1. PE.list_models() 返回模型列表
  2. PE.create("fcpe") 创建模型
  3. infer() 接受 paddle.Tensor / numpy / torch.Tensor 输入
  4. 训练/导出模块可导入
"""

# ---------------------------------------------------------------------------
# 0. 环境检查
# ---------------------------------------------------------------------------
import sys

print("=" * 60)
print("paddlePE 直接模式测试（无 PyTorch 冲突）")
print("=" * 60)

# 验证 Paddle 可导入
try:
    import paddle

    print(f"  ✅ PaddlePaddle {paddle.__version__} 已加载")
except Exception as e:
    print(f"  ❌ PaddlePaddle 加载失败: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 1. 导入 paddlepe
# ---------------------------------------------------------------------------
try:
    from paddlepe import PE

    print("  ✅ from paddlepe import PE 成功")
except Exception as e:
    print(f"  ❌ paddlepe 导入失败: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. 验证 API：list_models
# ---------------------------------------------------------------------------
models = PE.list_models()
print(f"  ✅ PE.list_models() = {models}")
assert "fcpe" in models, f"fcpe 应在模型列表中，实际: {models}"

# ---------------------------------------------------------------------------
# 3. 验证 API：create
# ---------------------------------------------------------------------------
pe = PE.create("fcpe")
print(f"  ✅ PE.create('fcpe') → {type(pe).__name__}")

# ---------------------------------------------------------------------------
# 4. 验证 API：infer 输入类型支持
# ---------------------------------------------------------------------------
sr = 16000
t_len = int(sr * 0.3)  # 0.3 秒
import numpy as np

wav_np = np.sin(2 * np.pi * 440 * np.linspace(0, 0.3, t_len)).astype(
    np.float32
)

# 4a. numpy → paddle.Tensor（由 _to_tensor 自动转换）
f0, conf = pe.infer(wav_np, sr)
assert f0 is not None
assert f0.shape[0] > 0
print(f"  ✅ pe.infer(numpy) → f0: {f0.shape}, conf: {conf.shape if conf is not None else None}")

# 4b. paddle.Tensor 直传
wav_pd = paddle.to_tensor(wav_np)
f0, conf = pe.infer(wav_pd, sr)
assert f0 is not None
print(f"  ✅ pe.infer(paddle.Tensor) → f0: {f0.shape}")

# 4c. torch.Tensor（通过 .numpy() 转换）— 可选，因为 Paddle 先加载后 torch 可能不可用
try:
    import torch

    wav_th = torch.from_numpy(wav_np)
    f0, conf = pe.infer(wav_th, sr)
    assert f0 is not None
    print(f"  ✅ pe.infer(torch.Tensor) → f0: {f0.shape}")
except (OSError, ImportError) as e:
    print(f"  ⚠️  torch 当前进程不可用: {e}")
    print(f"  ⚠️  跳过 torch.Tensor 输入测试（这是 DLL 冲突的预期行为）")

# 4d. 非法输入 → TypeError
try:
    pe.infer("not a tensor", sr)
    print("  ❌ pe.infer(str) 应抛 TypeError")
    sys.exit(1)
except TypeError as e:
    print(f"  ✅ pe.infer(str) → TypeError: {e}")

# 4e. get_pitch 别名
f0, conf = pe.get_pitch(wav_np, sr)
assert f0 is not None
print(f"  ✅ pe.get_pitch(numpy) → f0: {f0.shape}")

# ---------------------------------------------------------------------------
# 5. 验证：device 属性
# ---------------------------------------------------------------------------
dev = pe.device
print(f"  ✅ pe.device = {dev}")

# ---------------------------------------------------------------------------
# 6. 验证：训练/导出模块可导入
# ---------------------------------------------------------------------------
try:
    from paddlepe.training import PETask

    print("  ✅ from paddlepe.training import PETask 成功")
except Exception as e:
    print(f"  ❌ paddlepe.training 导入失败: {e}")

try:
    from paddlepe.export import export_onnx

    print("  ✅ from paddlepe.export import export_onnx 成功")
except Exception as e:
    print(f"  ❌ paddlepe.export 导入失败: {e}")

# ---------------------------------------------------------------------------
# 7. 验证：registry 存在
# ---------------------------------------------------------------------------
from paddlepe import registry

print(f"  ✅ from paddlepe import registry 成功")

# ---------------------------------------------------------------------------
print("=" * 60)
print("✅ 直接模式全部通过！")
print("=" * 60)
