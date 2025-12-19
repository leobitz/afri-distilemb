for dataset_name in "pos" "ner" 
do
    for emb_model_id in "leobitz/distil-emb-base" "leobitz/distil-emb-small-gelu"
    do
        for run_id in 1
        do
            for is_pretrained in 0
            do
                python train_token_classifier.py \
                    --distill_emb_model_id $emb_model_id \
                    --dataset_name $dataset_name \
                    --pretrained $is_pretrained \
                    --logging_step 100 \
                    --wandb_logging 1 \
                    --num_samples -1 \
                    --run_id $run_id \
                    --hidden_size 512 \
                    --num_hidden_layers 1
            done
        done
    done
done