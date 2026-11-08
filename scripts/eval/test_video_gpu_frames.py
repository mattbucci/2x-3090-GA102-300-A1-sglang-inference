#!/usr/bin/env python
"""Patch 077 unit test: video frames decoded on the GPU (torchcodec CUDA backend, the
default video load path since SGLang v0.5.23) must still reach the numpy-only consumers
(nano_nemotron_vl.parse_video, gemma4's frame sampler). One GPU, a few seconds.

    CUDA_VISIBLE_DEVICES=0 python scripts/eval/test_video_gpu_frames.py

Exit 1 on failure. Builds a synthetic 24-frame MP4 in memory (imageio/libx264 like
validate_capabilities.py), decodes it with VideoDecoderWrapper(device="cuda") and feeds
the frames through both consumers.
"""
import io
import sys

import numpy as np
import torch

from sglang.srt.utils.video_decoder import VideoDecoderWrapper, _try_cuda_backend


def synthetic_mp4(n=24, h=64, w=96) -> bytes:
    import imageio.v3 as iio

    frames = []
    for i in range(n):
        f = np.zeros((h, w, 3), dtype=np.uint8)
        f[:, :, 0] = (i * 10) % 256
        f[10:30, 10 + i : 30 + i, 1] = 255
        frames.append(f)
    buf = io.BytesIO()
    iio.imwrite(buf, frames, extension=".mp4", fps=12)
    return buf.getvalue()


def main() -> int:
    ok = True
    data = synthetic_mp4()
    cuda_ok = _try_cuda_backend()
    print(f"torchcodec cuda backend: {cuda_ok}")
    vdw = VideoDecoderWrapper(data, device="cuda" if cuda_ok else "cpu")
    idx = list(range(0, len(vdw), 4))
    frames = vdw.get_frames_at(idx)
    print(f"get_frames_at -> {type(frames).__name__}"
          f"{' cuda' if isinstance(frames, torch.Tensor) and frames.is_cuda else ''} shape={tuple(frames.shape)}")

    # consumer 1: nemotron parse_video (static) -> must return a numpy array
    from sglang.srt.multimodal.processors.nano_nemotron_vl import NanoNemotronVLImageProcessor as NN
    arr, ts = NN.parse_video(vdw)
    good = isinstance(arr, np.ndarray) and arr.ndim == 4 and arr.shape[-1] == 3 and len(ts) == arr.shape[0]
    print(f"nemotron parse_video -> {type(arr).__name__} {getattr(arr, 'shape', None)} timestamps={len(ts)} {'PASS' if good else 'FAIL'}")
    ok &= good
    # the exact call that 500'd on v0.5.23
    from PIL import Image
    try:
        Image.fromarray(arr[0], mode="RGB")
        print("Image.fromarray(frame) PASS")
    except Exception as e:  # noqa: BLE001
        print(f"Image.fromarray(frame) FAIL: {e}")
        ok = False

    # consumer 2: gemma4 frame sampler (method body re-run on a stub self)
    from sglang.srt.multimodal.processors import gemma4 as g4
    cls = [c for c in vars(g4).values() if isinstance(c, type) and hasattr(c, "_video_decoder_to_tensor")]
    if cls:
        stub = cls[0].__new__(cls[0])
        stub._processor = None
        try:
            t = cls[0]._video_decoder_to_tensor(stub, vdw)
            good = isinstance(t, torch.Tensor) and t.device.type == "cpu" and t.shape[1] == 3
            print(f"gemma4 _video_decoder_to_tensor -> {t.device} {tuple(t.shape)} {'PASS' if good else 'FAIL'}")
            ok &= good
        except Exception as e:  # noqa: BLE001
            print(f"gemma4 _video_decoder_to_tensor FAIL: {e}")
            ok = False
    else:
        print("gemma4 sampler not found (skipped)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
