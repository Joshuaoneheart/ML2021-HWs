#!/bin/bash
python3./fairseq/scripts/average_checkpoints.py --inputs ./checkpoints/rnn --num-epoch-checkpoints 5 --output ./checkpoint/rnn/average_last_5_checkpoint.pt
