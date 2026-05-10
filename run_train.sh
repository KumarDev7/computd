#!/bin/bash
cd /kaggle/working/computd
python -u scripts/train_tpu_1b.py 2>&1 | tee /kaggle/working/computd/train_1b_output.log