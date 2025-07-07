from datasets import load_dataset
import pandas as pd

langs = ['afaanoromoo', 'amharic', 'gahuza', 'hausa', 'igbo', 'pidgin', 'somali', 'swahili', 'tigrinya', 'yoruba']
all_train_texts = []
all_test_texts = []
lang2sentence_train = {k:[] for k in langs}
lang2sentence_test = {k:[] for k in langs}
for lang in langs:
    ds = load_dataset("castorini/afriberta-corpus", lang)
    df = pd.DataFrame(ds['train'])
    texts = df['text'].tolist()
    all_train_texts.extend(texts)
    lang2sentence_train[lang].extend(texts)
    df = pd.DataFrame(ds['test'])
    texts = df['text'].tolist()
    all_test_texts.extend(texts)
    lang2sentence_test[lang].extend(texts)
    

# with open('afriberta_train.txt', 'w') as f:
#     for text in all_train_texts:
#         f.write(text + '\n')
# with open('afriberta_test.txt', 'w') as f:
#     for text in all_test_texts:
#         f.write(text + '\n')

# save lang\ttext
with open('afriberta_train_lang.txt', 'w') as f:
    for lang in lang2sentence_train.keys():
        for text in lang2sentence_train[lang]:
            text = text.strip().replace('\n', ' ')
            line = f"{lang}\t{text}\n"
            f.write(line)

with open('afriberta_train_test.txt', 'w') as f:
    for lang in lang2sentence_test.keys():
        for text in lang2sentence_test[lang]:
            text = text.strip().replace('\n', ' ')
            line = f"{lang}\t{text}\n"
            f.write(line)