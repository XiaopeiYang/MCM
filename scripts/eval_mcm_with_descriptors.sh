EXP_NAME=$1
ID=$2
SCORE=$3
Des=$4

CKPT=ViT-B/16
DATA_ROOT=datasets
# Add --eval_with_description flag to the command


python eval_ood_detection.py --in_dataset ${ID} --name ${EXP_NAME} --CLIP_ckpt ${CKPT} --score ${SCORE} --root-dir ${DATA_ROOT} --eval_with_description ${Des}
