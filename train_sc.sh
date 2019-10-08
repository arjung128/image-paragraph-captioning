python train.py \
    --input_json data/paratalk.json \
    --input_fc_dir data/parabu_fc \
    --input_att_dir data/parabu_att \
    --input_label_h5 data/paratalk_label.h5 \
    --batch_size 10 \
    --learning_rate 5e-5 \
    --learning_rate_decay_start 0 \
    --scheduled_sampling_start 0 \
    --save_checkpoint_every 4 \
    --language_eval 1 \
    --val_images_use 5000 \
    --max_epochs 250 \
    --self_critical_after 0 \
    --cached_tokens para_train-idxs \
    --cider_reward_weight 1 \
    --block_trigrams 1 \
    --start_from "alpha=0_log_sc" \
    --checkpoint_path "alpha=0_log_sc" \
    --id 'xe' \
    --print_freq 200
# removed --caption_model topdown \
# added --id 'xe' \
# changed --cached_tokens para-train-idxs to --cached_tokens para_train-idxs
