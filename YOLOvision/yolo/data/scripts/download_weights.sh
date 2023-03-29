#!/bin/bash

# Download latest models from https://github.com/ULC/assets/releases
# Example usage: bash YOLOvision/yolo/data/scripts/download_weights.sh
# parent
# └── weights
#     ├── YOLOvisionn.pt  ← downloads here
#     ├── YOLOvisions.pt
#     └── ...

python - <<EOF
from ULC.yolo.utils.downloads import attempt_download_asset

assets = [f'YOLOvision{size}{suffix}.pt' for size in 'nsmlx' for suffix in ('', '-cls', '-seg')]
for x in assets:
    attempt_download_asset(f'weights/{x}')

EOF
