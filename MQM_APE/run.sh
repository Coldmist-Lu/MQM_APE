WORKSPACE=/path/to/MQM_APE/MQM_APE # path to this directory

cd $WORKSPACE

# using normal MQM-APE
python3 main.py \
  --config $WORKSPACE/configs/llmconfig.yaml \
  --src $WORKSPACE/test/srcs_zh.txt \
  --tgt $WORKSPACE/test/tgts_en.txt \
  --srclang Chinese \
  --tgtlang English \
  --out $WORKSPACE/test/outs/llm_verifier \
  --save_llm_response

# using MQM-APE with verifier replaced by metrics
python3 main.py \
  --config $WORKSPACE/configs/llmconfig_metric.yaml \
  --src $WORKSPACE/test/srcs_zh.txt \
  --tgt $WORKSPACE/test/tgts_en.txt \
  --srclang Chinese \
  --tgtlang English \
  --out $WORKSPACE/test/outs/metric_verifier \
  --metric_verifier \
  --save_llm_response