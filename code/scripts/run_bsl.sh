# baseline
for _datamode in cv fb food; do
    python main.py \
        --use_cache 0  \
        --plm_lr 3e-5 \
        --lr 3e-5 \
        --plm "bart-base" \
        --batch_size 16 \
        --eval_batch_size 16 \
        --no_dl_score 0 \
        --patience 15 \
        --num_epochs 25 \
        --num_evals_per_epoch 2 \
        --exp_msg "bsl"  \
        --data_mode $_datamode \
        --pred_with_gen 1
done