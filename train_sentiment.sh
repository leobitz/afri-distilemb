python train_classifier.py \
    --distill_emb_model_id "leobitz/distil-emb-base" \
    --dataset_name "hate" \
    --pretrained 1 \
    --logging_step 100 \
    --wandb_logging 1 \
    --num_samples -1 \
    --run_id 0 \
    --hidden_size 786 \
    --num_hidden_layers 3

python train_classifier.py \
    --distill_emb_model_id "leobitz/distil-emb-base" \
    --dataset_name "sentiment" \
    --pretrained 1 \
    --logging_step 100 \
    --wandb_logging 1 \
    --num_samples -1 \
    --run_id 0 \
    --hidden_size 786 \
    --num_hidden_layers 3

python train_classifier.py \
    --distill_emb_model_id "leobitz/distil-emb-base" \
    --dataset_name "news" \
    --pretrained 1 \
    --logging_step 100 \
    --wandb_logging 1 \
    --num_samples -1 \
    --run_id 0 \
    --hidden_size 786 \
    --num_hidden_layers 3