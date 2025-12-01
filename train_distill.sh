python train_distill.py --max_epochs 512 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
    --dropout 0.1 --normalize --use_tanh --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
    --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
    --temperature 0.1 --seq_len 64 --max_word_piece 10