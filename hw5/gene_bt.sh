 !/bin/bash
binpath="./DATA/data-bin/synthetic"
src_dict_file="./DATA/data-bin/ted2020/dict.en.txt"
tgt_dict_file=$src_dict_file
monopref="./DATA/rawdata/mono/mono.tok"
python -m fairseq_cli.preprocess\
         --source-lang 'zh'\
          --target-lang 'en'\
          --trainpref $monopref\
          --destdir $binpath\
          --srcdict $src_dict_file\
          --tgtdict $tgt_dict_file\
          --workers 2
cp -r ./DATA/data-bin/ted2020/ ./DATA/data-bin/ted2020_with_mono/

cp ./DATA/data-bin/synthetic/train.zh-en.zh.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.bin
cp ./DATA/data-bin/synthetic/train.zh-en.zh.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.zh.idx
cp ./DATA/data-bin/synthetic/train.zh-en.en.bin ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.bin
cp ./DATA/data-bin/synthetic/train.zh-en.en.idx ./DATA/data-bin/ted2020_with_mono/train1.en-zh.en.idx
