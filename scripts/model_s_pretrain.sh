BERT_BASE_DIR="bert_model_cn/model_s_chinese_L-12_H-768_A-12"

python run_pretraining.py \
  --input_file=pretrain_cn_output_model_s/train.tfrecord \
  --output_dir=pretrain_cn_output_model_s/ \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  #--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=124000 \
  --num_warmup_steps=6200 \
  --learning_rate=2e-5