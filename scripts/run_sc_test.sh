import os
os.environ['BERT_BASE_DIR'] = "bert_model_cn/chinese_L-12_H-768_A-12"
os.environ['GLUE_DIR'] = "glue/glue_data"

!python3 run_classifier.py \
  --task_name=SC \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$GLUE_DIR/SC \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=4 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=trained_cn_output/