#!/usr/bin/env python
"""通过 paddlePE server 远程推理示例。

用法：
    # 1. 先在一个终端启动 server：
    #    start /B python -m paddlepe.server --model fcpe --port 28789
    #
    # 2. 再运行本脚本：
    #    python scripts/remote_infer.py
    #
    # 可选参数：
    #    python scripts/remote_infer.py --port 28789 --input 其他音频.wav

服务器端点和说明：
    GET  /health              → 健康检查
    GET  /models              → 列出可用模型
    POST /load                → 加载模型 {"model": "fcpe"}
    POST /infer               → 推理（POST body = WAV 二进制）
    POST /shutdown            → 关闭服务器
"""

import argparse
import json
import os
import struct
import sys
import time
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import soundfile as sf

# 确保 paddlepe 可导入
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR.parent))

from paddlepe.io.formats import HEADER_SIZE, decode_header  # noqa: E402


def wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """Convert float32 audio to WAV bytes."""
    wav_int16 = (wav * 32767).clip(-32768, 32767).astype(np.int16)
    data_size = len(wav_int16) * 2
    buf = BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(wav_int16.tobytes())
    return buf.getvalue()


def parse_f0_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse .f0 binary response."""
    header = decode_header(data[:HEADER_SIZE])
    f0 = np.frombuffer(
        data,
        dtype=np.float32,
        count=header.num_frames,
        offset=header.header_size,
    ).copy()
    conf = None
    if header.flags & 0x01:
        conf = np.frombuffer(
            data,
            dtype=np.float32,
            count=header.num_frames,
            offset=header.header_size + header.num_frames * 4,
        ).copy()
    return f0, conf


def main():
    parser = argparse.ArgumentParser(description="paddlePE 远程推理客户端")
    parser.add_argument("--port", type=int, default=28789, help="server 端口")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="输入音频路径（默认 scripts/example_song.flac）",
    )
    parser.add_argument(
        "--model", type=str, default="fcpe", help="模型名（需 server 已加载）"
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="有声/无声阈值")
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"

    # ── 检查 server 是否在线 ──
    print(f"[client] 连接 server {base_url} ...")
    try:
        resp = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        info = json.loads(resp.read())
        print(f"[client] ✅ server 在线: {info}")
    except Exception as e:
        print(f"[client] ❌ 无法连接 server ({e})")
        print(
            f"[client] 请先启动: start /B python -m paddlepe.server "
            f"--model {args.model} --port {args.port}"
        )
        return

    # ── 加载音频 ──
    wav_path = args.input or str(_THIS_DIR / "example_song.flac")
    if not os.path.exists(wav_path):
        print(f"[client] ❌ 找不到音频: {wav_path}")
        return

    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    duration = len(wav) / sr
    print(f"[client] 音频: {wav_path}")
    print(f"[client] 时长: {duration:.1f}s, 采样率: {sr}Hz")

    # ── 推理 ──
    wav_bytes = wav_to_bytes(wav, sr)
    headers = {
        "Content-Type": "audio/wav",
        "X-Threshold": str(args.threshold),
    }
    req = urllib.request.Request(f"{base_url}/infer", data=wav_bytes, headers=headers)

    print(f"[client] 发送推理请求 ({len(wav_bytes)} bytes)...")
    t0 = time.time()
    resp = urllib.request.urlopen(req, timeout=300)
    f0_bytes = resp.read()
    elapsed = time.time() - t0

    f0, conf = parse_f0_bytes(f0_bytes)
    valid = f0 > 0
    rtf = elapsed / duration

    print(f"[client] 推理耗时: {elapsed:.3f}s")
    print(f"[client] 实时率 (RTF): {rtf:.3f}x ({'实时' if rtf < 1 else '慢于实时'})")
    print(f"[client] 帧数: {len(f0)}, 有声: {valid.sum()}/{len(f0)}")
    if valid.any():
        print(f"[client] F0 范围: [{f0[valid].min():.0f}, {f0[valid].max():.0f}] Hz")
    print("[client] ✅ 推理完成")


if __name__ == "__main__":
    main()
