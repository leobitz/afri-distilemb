from datasets import load_dataset
import pandas as pd

langs = ['afaanoromoo', 'amharic', 'gahuza', 'hausa', 'igbo', 'pidgin', 'somali', 'swahili', 'tigrinya', 'yoruba']
all_train_texts = []
all_test_texts = []
for lang in langs:
    ds = load_dataset("castorini/afriberta-corpus", lang)
    df = pd.DataFrame(ds['train'])
    texts = df['text'].tolist()
    all_train_texts.extend(texts)
    df = pd.DataFrame(ds['test'])
    texts = df['text'].tolist()
    all_test_texts.extend(texts)

with open('afriberta_train.txt', 'w') as f:
    for text in all_train_texts:
        f.write(text + '\n')
with open('afriberta_test.txt', 'w') as f:
    for text in all_test_texts:
        f.write(text + '\n')