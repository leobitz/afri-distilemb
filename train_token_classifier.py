from modeling_distillemb import BertForTokenClassification
from distill_emb import DistillEmb
from config import DistillModelConfig, DistillEmbConfig
import torch
from tokenizer import CharTokenizer
from data_loader import load_ner_dataset, load_pos_dataset
import os
from datasets import Dataset
from typing import Dict, Any
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import os
import argparse
from data_loader import *
from helper import anonymize_and_normalize_text
from huggingface_hub import snapshot_download
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("--distill_emb_model_id", type=str, required=True,
                    help="Path or name of the DistillEmb model.")
parser.add_argument("--dataset_name", type=str, required=True,
                    help="Dataset name (used to load parquet file).")

parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_hidden_layers", type=int, default=1)
parser.add_argument("--hidden_dropout_prob", type=float, default=0.5)
parser.add_argument("--max_seq_length", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--label_smoothing_factor", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=5.0)
parser.add_argument("--warmup_ratio", type=float, default=0.0)
parser.add_argument("--grad_accumulation_steps", type=int, default=1)
parser.add_argument("--pretrained", type=int, default=1)
parser.add_argument("--logging_step", type=int, default=1)
parser.add_argument("--wandb_logging", type=int, default=1)
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("--run_id", type=str, default=None)



args = parser.parse_args()

distill_emb_model_id = args.distill_emb_model_id
dataset_name = args.dataset_name
hidden_size = args.hidden_size
num_hidden_layers = args.num_hidden_layers
hidden_dropout_prob = args.hidden_dropout_prob
max_seq_length = args.max_seq_length
batch_size = args.batch_size
learning_rate = args.learning_rate
num_train_epochs = args.num_train_epochs
weight_decay = args.weight_decay
label_smoothing_factor = args.label_smoothing_factor
max_grad_norm = args.max_grad_norm
warmup_ratio = args.warmup_ratio
grad_accumulation_steps = args.grad_accumulation_steps
pretrained_distill = bool(args.pretrained)
logging_step = args.logging_step
wandb_logging = bool(args.wandb_logging)
num_samples = args.num_samples
run_id = args.run_id 

is_pretrained = "pretrained" if pretrained_distill else "scratch"
run_name = f"{dataset_name}_{distill_emb_model_id.replace('/', '_')}_hs{hidden_size}_layers{num_hidden_layers}_{is_pretrained}_{run_id}"
if wandb_logging:
    os.environ["WANDB_PROJECT"] = "distillemb"

model_dir = snapshot_download(distill_emb_model_id, local_dir=f"./pretrained_models/{distill_emb_model_id}")
distill_model = DistillEmb.from_pretrained(pretrained_model_name_or_path=model_dir)
tokenizer = CharTokenizer.from_pretrained(pretrained_directory=model_dir)
distill_config = DistillEmbConfig.from_pretrained(pretrained_model_name_or_path=model_dir)

config = DistillModelConfig(
    hidden_size=hidden_size,
    num_hidden_layers=num_hidden_layers,
    hidden_dropout_prob=hidden_dropout_prob,
    embedding_type="distill",  # 'distilemb', 'fasttext'
    encoder_type='lstm', #'lstm'
    char_vocab_size=tokenizer.char_vocab_size,
    distill_config=distill_config,
    distill_pretrained_model_name=distill_emb_model_id if pretrained_distill else None
)


if dataset_name == 'pos':
    df, labels = load_pos_dataset()
elif dataset_name == 'ner':
    df, labels = load_ner_dataset()
else:
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")

labels = list(range(max(labels) + 1))
df['text'] = df['tokens'].apply(lambda x: ' '.join(x))
df = df[df['text'].str.strip().astype(bool)].sample(frac=1.0, random_state=42).reset_index(drop=True)

if num_samples > 0:
    df = df.sample(num_samples, random_state=42).reset_index(drop=True)

assert 'text' in df.columns, f"Dataframe must contain a 'text' column, found columns: {df.columns}"
assert 'labels' in df.columns, f"Dataframe must contain a 'labels' column, found columns: {df.columns}"
assert 'split' in df.columns, f"Dataframe must contain a 'split' column, found columns: {df.columns}"
assert 'lang' in df.columns, f"Dataframe must contain a 'lang' column, found columns: {df.columns}"

lang_counts = df.groupby('split')['lang'].nunique()
for split, count in lang_counts.items():
    print(f"{split.capitalize()} split has {count} languages.")

label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
config.label2id = label2id
config.id2label = id2label

print(f"Converted labels to integers: {label2id}")
print(f"Converted integers to labels: {id2label}")

config.num_labels = len(labels)
model = BertForTokenClassification(config)

text_col = 'text'

df['text'] = df[text_col].apply(anonymize_and_normalize_text)
train_df = df[df['split'] == 'train'][['text', 'labels']]
test_df = df[df['split'] == 'test'][['text', 'labels']]

# Create HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


def preprocess_function(examples: Dict[str, Any]):
    batch = tokenizer(
        examples["text"],
        padding=False,
        max_length=max_seq_length,
        return_attention_mask=False,
    )

    batch["labels"] = examples["labels"]
    return batch

tokenized_train = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

tokenized_test = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names,
)

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            max_length=max_seq_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        max_len = batch["input_ids"].shape[1] - 2  # exclude special tokens
        padded_labels = []
        for f in features:
            label = f["labels"]
            
            padded_label = [-100] +  label + [-100] * (max_len - len(label)) + [-100]
            padded_labels.append(padded_label)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        assert batch["labels"].shape == (batch["input_ids"].shape[0], batch["input_ids"].shape[1]), f"Labels shape {batch['labels'].shape} does not match input_ids shape {batch['input_ids'].shape}"
        return batch
    

data_collator = CustomDataCollator(tokenizer)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        mask = label_seq != -100
        true_labels.extend(label_seq[mask])
        pred_labels.extend(pred_seq[mask])

    label_ids = list(label2id.values())

    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_weighted": f1_score(true_labels, pred_labels, average="weighted", labels=label_ids, zero_division=0),
        "f1_macro": f1_score(true_labels, pred_labels, average="macro", labels=label_ids, zero_division=0),
        "f1_micro": f1_score(true_labels, pred_labels, average="micro", labels=label_ids, zero_division=0),
    }

dataloader_num_workers=os.cpu_count() - 1

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_train_epochs,
    weight_decay=weight_decay,
    report_to=["wandb"] if wandb_logging else [],
    eval_strategy="epoch",  
    logging_strategy="steps",
    logging_steps=logging_step,
    label_smoothing_factor=label_smoothing_factor,
    max_grad_norm=max_grad_norm,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type="cosine",
    dataloader_num_workers=dataloader_num_workers,        # Number of CPU workers for data loading
    dataloader_pin_memory=True,      # Faster GPU transfer
    gradient_accumulation_steps=grad_accumulation_steps,
    run_name=run_name,
    project="distillemb"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


trainer.train()


model = trainer.model
model.eval()

test_df = df[df['split'] == 'test'][['text', 'label', 'lang']]
languages = test_df['lang'].unique()
per_language_f1 = {}
all_preds = []
all_labels = []
for lang in languages:
    lang_df = test_df[test_df['lang'] == lang]
    texts = lang_df['text'].tolist()
    labels = lang_df['label'].values
    preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]
        tokenized = tokenizer(
            batch_texts,
            padding='longest',
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_attention_mask=True,
            padding_side="right"
        )
        with torch.no_grad():
            inputs = {k: v.cuda() for k, v in tokenized.items()}
            outputs = model(**inputs)
            batch_preds = outputs.logits.cpu().numpy()
            preds.append(batch_preds)
    preds = np.vstack(preds)
    evals = compute_metrics((preds, labels))
    per_language_f1[lang] = evals
    all_preds.extend(preds)
    all_labels.extend(labels)

all_preds = np.vstack(all_preds)
all_labels = np.array(all_labels)
print(all_preds.shape, all_labels.shape)
all_eval = compute_metrics((all_preds, all_labels))
per_language_f1['all'] = all_eval

metric_keys = next(iter(per_language_f1.values())).keys()
avg_metrics = {}
for metric in metric_keys:
    vals = [scores[metric] for lang, scores in per_language_f1.items() if lang != 'all']
    avg_metrics[metric] = float(np.mean(vals))
per_language_f1['avg'] = avg_metrics


if len(all_labels) > 0:
    all_preds = np.vstack(all_preds)
    all_labels = np.array(all_labels)
    print(all_preds.shape, all_labels.shape)
    all_eval = compute_metrics((all_preds, all_labels))
    per_language_f1['ood-lang'] = all_eval

df = pd.DataFrame.from_dict(per_language_f1, orient='index')
# create eval_results directory if it doesn't exist
os.makedirs(f'eval_results/{dataset_name}', exist_ok=True)
df.to_csv(f'eval_results/{dataset_name}/{run_name}.csv')