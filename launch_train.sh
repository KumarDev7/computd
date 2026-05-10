#!/bin/bash
cd /kaggle/working/computd
exec python -u scripts/train_tpu_1b.py 2>&1
