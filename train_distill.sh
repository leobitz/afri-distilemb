# python train_distill.py --max_epochs 28 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
#     --dropout 0.1 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-config --activation gelu \
#     --run_name base-config --size base
#     # --normalize --use_tanh 

# python train_distill.py --max_epochs 28 --batch_size 2048 --lr 0.0001 --weight_decay 0.00 \
#     --dropout 0.1 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-config --activation gelu \
#     --run_name base-normalize --size base --normalize

# python train_distill.py --max_epochs 28 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
#     --dropout 0.1 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-config --activation gelu \
#     --run_name base-norm --size base --use_normalize

# python train_distill.py --max_epochs 28 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
#     --dropout 0.1 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-config --activation gelu \
#     --run_name base-tanh --size base --use_tanh
#     # --normalize --use_tanh 

# python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base --activation gelu \
#     --run_name base-drop0.0 --size base

# python train_distill.py --max_epochs 128 --batch_size 512 --lr 0.001 --weight_decay 0.0 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-b512 --activation gelu \
#     --run_name base-drop0.0 --size base

# python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-silu --activation silu \
#     --run_name base-silu --size base

    # python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
    # --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
    # --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
    # --temperature 0.1 --seq_len 64 --max_word_piece 10  \
    # --hf_repo_id leobitz/distil-emb-base-tanh --activation tanh \
    # --run_name base-tanh --size base

python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
    --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
    --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
    --temperature 0.1 --seq_len 64 --max_word_piece 10 \
    --hf_repo_id leobitz/distil-emb-small-leaky_relu --activation leaky_relu \
    --run_name base-leaky_relu --size small

python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
    --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
    --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
    --temperature 0.1 --seq_len 64 --max_word_piece 10 \
    --hf_repo_id leobitz/distil-emb-small-relu --activation relu \
    --run_name base-relu --size small

python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.0 \
    --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
    --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
    --temperature 0.1 --seq_len 64 --max_word_piece 10  --size small \
    --hf_repo_id leobitz/distil-emb-small-gelu --activation gelu \
    --run_name base-gelu --size small
# python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.05 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-wd05 --activation gelu \
#     --run_name base-drop0.0-wd0.05 --size base  --use_normalize

# python train_distill.py --max_epochs 28 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.07 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-config --activation gelu \
#     --run_name base-temp0.07 --size base

# python train_distill.py --max_epochs 128 --batch_size 2048 --lr 0.001 --weight_decay 0.00 \
#     --dropout 0.0 --clip_grad_norm 1.0 --vector_load_ratio 1.0 \
#     --sentence_load_ratio 1.0 --min_sent_length 16 --train_ratio 0.99 --neg_seq_len 64 \
#     --temperature 0.1 --seq_len 64 --max_word_piece 10  \
#     --hf_repo_id leobitz/distil-emb-base-relu --activation relu \
#     --run_name base-relu --size base  --use_normalize