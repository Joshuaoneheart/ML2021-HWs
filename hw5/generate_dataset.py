import sentencepiece as spm
spm_model = spm.SentencePieceProcessor(model_file=str('DATA/rawdata/ted2020/spm8000.model'))
in_tag = f"./DATA/rawdata/mono/mono.predict.en"
split, lang = 'mono.tok', 'en'
out_path = f'./DATA/rawdata/mono/{split}.{lang}'
with open(out_path, 'w') as out_f:
    with open(in_tag, 'r') as in_f:
        for line in in_f:
            line = line.strip()
            tok = spm_model.encode(line, out_type=str)
            print(' '.join(tok), file=out_f)
