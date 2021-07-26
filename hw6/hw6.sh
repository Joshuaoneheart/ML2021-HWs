#!/bin/bash
workspace_dir = '.'
data_path=$1
unzip -q $datapath -d .
pip install stylegan2_pytorch pytorch-fid
 
# stylegan2_pytorch --data ./faces --multi-gpus --results_dir ./result --models_dir ./models --image_size=64 --network-capacity 64 --num-train-steps 3000
stylegan2_pytorch --generate --num_generate=1000 --image_size=64 --num_image_tiles 1
 
i=1
for f in `ls results/default/*-ema.jpg` ; do cp $f $i.jpg; ((i++)); done
ls *.jpg | wc -l
tar -zcf ./images.tgz *.jpg
rm *.jpg

