export CUDA_VISIBLE_DEVICES=0
# This my data folder
# aevnmt/demo$ ls data.en/
# ptb  quora  short_yelp  snli  ted  tedenes  tedentr  tedentrsmall  tednewsmonoen  tednewsmonoendomain  tednewsmonoenes  yahoo
USE_GPU=true
LANG=en
TAG=ptb
DATA=data.${LANG}/${TAG}
OUTPUT=senvae-models/${LANG}/${TAG}/1
HPARAMS=hparams/senvae_gaussian.json

mkdir -p ${OUTPUT}

python -m aevnmt.senvae \
    --src ${LANG} \
    --tgt ${LANG} \
    --validation_prefix ${DATA}/valid \
    --mono_src ${DATA}/train.${LANG} \
    --output_dir ${OUTPUT} \
    --hparams_file ${HPARAMS} \
    --use_gpu ${USE_GPU} 
