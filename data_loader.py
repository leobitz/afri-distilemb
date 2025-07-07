import os
import pandas as pd
from datasets import load_dataset

data_path = "downstream-data"

def load_news_dataset():
    if not os.path.exists(f'{data_path}/masakhanews.parquet'):
        langs = ['amh', 'eng', 'fra', 'hau', 'ibo', 'lin', 'lug', 'orm', 'pcm', 'run', 'sna', 'som', 'swa', 'tir', 'xho', 'yor']
        dss = {}
        for lang in langs:
            data = load_dataset('masakhane/masakhanews', lang, trust_remote_code=True) 
            dss[lang] = data
        # Concatenate all datasets
        # train_data = pd.concat([dss[lang]['train'].to_pandas() for lang in dss.keys()], ignore_index=True)
        # test_data = pd.concat([data['test'].to_pandas() for lang in dss.keys()], ignore_index=True)
        # dev_data = pd.concat([data['validation'].to_pandas() for lang in dss.keys()], ignore_index=True)
        train_data = []
        test_data = []
        dev_data = []
        for lang, data in dss.items():
            # convert to pandas DataFrame
            train_data.append(data['train'].to_pandas())
            test_data.append(data['test'].to_pandas())
            dev_data.append(data['validation'].to_pandas())
            # set the language column
            train_data[-1]['lang'] = lang
            test_data[-1]['lang'] = lang
            dev_data[-1]['lang'] = lang
        train_data = pd.concat(train_data, ignore_index=True)
        test_data = pd.concat(test_data, ignore_index=True)
        dev_data = pd.concat(dev_data, ignore_index=True)

        # Add split column
        train_data['split'] = 'train'
        test_data['split'] = 'test'
        dev_data['split'] = 'dev'
        # Concatenate all data
        all_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)
        # apply strip to text
        all_data['text'] = all_data['text'].str.strip()
        # remove empty and null texts
        all_data = all_data[all_data['text'].notnull() & (all_data['text'] != '')]
        # remove duplicates
        all_data = all_data.drop_duplicates(subset=['text'])
        print(f'Loaded {len(all_data)} rows from masakhanews columns {all_data.columns}')
        # save to parquet
        all_data.to_parquet(f'{data_path}/masakhanews.parquet', index=False)
    else:
        all_data = pd.read_parquet(f'{data_path}/masakhanews.parquet')
    print(f'Loaded {len(all_data)} rows from masakhanews.parquet columns {all_data.columns}')
    unque_labels = set(all_data['label'].unique())
    return all_data, sorted(unque_labels)

def load_ner_dataset():
    if not os.path.exists(f'{data_path}/masakhanener.parquet'):
        langs = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']
        dss = []
        for lang in langs:
            data = load_dataset('masakhane/masakhaner2', lang, trust_remote_code=True) 
            dss.append(data)
        # Concatenate all datasets
        train_data = pd.concat([data['train'].to_pandas() for data in dss], ignore_index=True)
        test_data = pd.concat([data['test'].to_pandas() for data in dss], ignore_index=True)
        dev_data = pd.concat([data['validation'].to_pandas() for data in dss], ignore_index=True)
        # Add split column
        train_data['split'] = 'train'
        test_data['split'] = 'test'
        dev_data['split'] = 'dev'
        # Concatenate all data
        all_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)
        # rename ner_tags column to labels
        all_data.rename(columns={'ner_tags': 'labels'}, inplace=True)
        print(f'Loaded {len(all_data)} rows from masakhanener columns {all_data.columns}')
        # save to parquet
        all_data.to_parquet('masakhanener.parquet', index=False)
    else:
        all_data = pd.read_parquet(f'{data_path}/masakhanener.parquet')
        print(f'Loaded {len(all_data)} rows from masakhanener.parquet columns {all_data.columns}')
    # from labels column, get unique labels
    unique_labels = set()
    for labels in all_data['labels']:
        unique_labels.update(labels)
    return all_data, sorted(unique_labels)

def load_pos_dataset():
    if not os.path.exists(f'{data_path}/masakhapos.parquet'):
        langs = ['bam', 'bbj', 'ewe', 'fon', 'hau', 'ibo', 'kin', 'lug', 'luo', 'mos', 'nya', 'pcm', 'sna', 'swa', 'tsn', 'twi', 'wol', 'xho', 'yor', 'zul']
        dss = []
        for lang in langs:
            data = load_dataset('masakhane/masakhapos', lang, trust_remote_code=True) 
            dss.append(data)
        # Concatenate all datasets
        train_data = pd.concat([data['train'].to_pandas() for data in dss], ignore_index=True)
        test_data = pd.concat([data['test'].to_pandas() for data in dss], ignore_index=True)
        dev_data = pd.concat([data['validation'].to_pandas() for data in dss], ignore_index=True)
        # Add split column
        train_data['split'] = 'train'
        test_data['split'] = 'test'
        dev_data['split'] = 'dev'
        # Concatenate all data
        all_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)
        # rename upos column to labels
        all_data.rename(columns={'upos': 'labels'}, inplace=True)
        print(f'Loaded {len(all_data)} rows from masakhapos columns {all_data.columns}')
        # save to parquet
        all_data.to_parquet(f'{data_path}/masakhapos.parquet', index=False)
    else:
        all_data = pd.read_parquet(f'{data_path}/masakhapos.parquet')
        print(f'Loaded {len(all_data)} rows from masakhapos.parquet columns {all_data.columns}')
    # from labels column, get unique labels
    unique_labels = set()
    for labels in all_data['labels']:
        unique_labels.update(labels)
    return all_data, sorted(unique_labels)

def load_sentiment_file(path, lang):
    lines = open(path, 'r', encoding='utf-8').read().strip().split('\n')
    data = []
    for line in lines:
        if line.strip():
            parts = line.split('\t')
            line_id, text, label = parts
            data.append({
                'id': line_id,
                'text': text,
                'label': label,
                'lang': lang
            })
    return data

def load_sentiment_dir(path):
    file_names = os.listdir(path)
    # ignore files other than tsv
    file_names = [f for f in file_names if f.endswith('.tsv')]
    # langs are part of the file name before _
    langs = sorted([f.split('_')[0] for f in file_names])
    data = []
    for file_name, lang in zip(file_names, langs):
        file_path = os.path.join(path, file_name)
        data.extend(load_sentiment_file(file_path, lang))
    return data

def load_sentiment_task_a(path):
    train_dir = path + '/train'
    test_dir = path + '/test'
    dev_dir = path + '/dev'

    train_data = load_sentiment_dir(train_dir)
    test_data = load_sentiment_dir(test_dir)
    dev_data = load_sentiment_dir(dev_dir)

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    dev_df = pd.DataFrame(dev_data)

    train_df['split'] = 'train'
    test_df['split'] = 'test'
    dev_df['split'] = 'dev'

    all_data = pd.concat([train_df, test_df, dev_df], ignore_index=True)
    return all_data

def load_sentiment_task_c(path):
    test_dir = path + '/test'
    dev_dir = path + '/dev'

    test_data = load_sentiment_dir(test_dir)
    dev_data = load_sentiment_dir(dev_dir)

    test_df = pd.DataFrame(test_data)
    dev_df = pd.DataFrame(dev_data)

    test_df['split'] = 'test'
    dev_df['split'] = 'test'

    all_data = pd.concat([test_df, dev_df], ignore_index=True)
    return all_data

def load_sentiment():
    path = f"{data_path}/afrisent-semeval-2023"
    if not os.path.exists(f"{data_path}/sentiment.parquet"):
        dfa = load_sentiment_task_a(f'{path}/SubtaskA')
        dfc = load_sentiment_task_c(f'{path}/SubtaskC')
        df = pd.concat([dfa, dfc], ignore_index=True)
        # remove id column
        df = df.drop(columns=['id'])
        # rmeove text column duplicates
        df = df.drop_duplicates(subset=['text'])
        # apply strip to text
        df['text'] = df['text'].str.strip()
        # remove empty and null texts
        df = df[df['text'].notnull() & (df['text'] != '')]
        print(f'Loaded {len(df)} rows from {path}')
        # remove duplicates
        # save to parquet
        df.to_parquet(f"{data_path}/sentiment.parquet", index=False)
    else:
        df = pd.read_parquet(f"{data_path}/sentiment.parquet")
        print(f'Loaded {len(df)} rows from sentiment.parquet columns {df.columns}')
    # remove rows with label == label
    df = df[df['label'] != 'label']
    unique_labels = set(df['label'].unique())
    # print(f'Unique labels: {unique_labels}')
    return df, sorted(unique_labels)




