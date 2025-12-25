for dataset_name in "ner" 
do
    for emb_model_id in "leobitz/distil-emb-base"
    do
        for run_id in 1
        do
            for is_pretrained in 1 
            do
                python train_token_classifier.py \
                    --distill_emb_model_id $emb_model_id \
                    --dataset_name $dataset_name \
                    --pretrained $is_pretrained \
                    --logging_step 100 \
                    --wandb_logging 1 \
                    --num_samples -1 \
                    --run_id $run_id \
                    --hidden_size 768 \
                    --num_hidden_layers 3 \
                    --hidden_dropout_prob 0.1 \
                    --num_samples 50
            done
        done
    done
done