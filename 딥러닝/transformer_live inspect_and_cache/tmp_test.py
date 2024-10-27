print(len(en2fr.source_sentences))
print(len(en2fr.target_sentences))

for src, tgt in zip(en2fr.source_sentences[1000:1010], en2fr.target_sentences[1000:1010]):
    print(f'Source: {src}')
    print(f'Target: {tgt}')

src_vocab = en2fr.source_vocab
tgt_vocab = en2fr.target_vocab

print(en2fr.source_vocab.oov_idx)
print(en2fr.source_vocab.pad_idx)
print(en2fr.target_vocab.pad_idx) 

print(en2fr.source_vocab.vocab_size) 
print(en2fr.target_vocab.vocab_size)

for idx, (batch_x, batch_y) in enumerate(en2fr.data):
    print(batch_x.shape)
    print(batch_y.shape)

    src, tgt = batch_x[0], batch_y[0]
    print(src)
    print(src_vocab.tensor2sentence(src, clean = True))
    print(tgt_vocab.tensor2sentence(tgt, clean = True))

    if idx > 1:
        break 
    
