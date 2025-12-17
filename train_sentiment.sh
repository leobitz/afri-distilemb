python train_classifier.py \
    --distill_emb_model_id "leobitz/distil-emb-base" \
    --dataset_name "hate" \
    --pretrained 1 \
    --logging_step 100 \
    --wandb_logging 1 \
    --num_samples 50