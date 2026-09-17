import torch
import os
from pathlib import Path

print("GPU Visible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

adapter = Path('/app/checkpoints/track_b_v2/adapter_final/adapter_model.safetensors')
print("Adapter Exists:", adapter.exists())
if adapter.exists():
    from satquery.tools.sidecars import readable_safetensors
    print("Adapter Readable:", readable_safetensors(adapter))

from satquery.tools.rs_vqa import is_available, TOOL_VERSION
print("is_available:", is_available())
print("version:", TOOL_VERSION)
