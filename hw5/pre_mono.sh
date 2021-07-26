#!/bin/bash
binpath="./DATA/data-bin/mono"
src_dict_file='./DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file=$src_dict_file
monopref="./DATA/rawdata/mono/mono.tok" # whatever filepath you get after applying subword tokenization
python -m fairseq_cli.preprocess\
	--source-lang 'zh'\
	--target-lang 'en'\
	--trainpref ${monopref}\
	--destdir ${binpath}\
	--srcdict ${src_dict_file}\
	--tgtdict ${tgt_dict_file}\
	--workers 2
cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin
cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx
cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin
cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx
