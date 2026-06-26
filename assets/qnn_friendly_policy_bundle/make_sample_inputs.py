#!/usr/bin/env python3
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / "sample_inputs"
root.mkdir(parents=True, exist_ok=True)
(root / "image.raw").write_bytes(np.zeros((1, 3, 480, 640), dtype=np.float32).tobytes())
(root / "qpos.raw").write_bytes(np.zeros((1, 14), dtype=np.float32).tobytes())
(root / "task_embedding.raw").write_bytes(np.zeros((1, 32), dtype=np.float32).tobytes())
(root / "input_list.txt").write_text(
    f"image:={root / 'image.raw'} qpos:={root / 'qpos.raw'} task_embedding:={root / 'task_embedding.raw'}\n"
)
print(root / "input_list.txt")
