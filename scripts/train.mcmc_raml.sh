#!/bin/sh

source ~/.bashrc
source activate code_mining_pytorch0.4

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
# train_src="data/valid.de-en.de"
# train_tgt="data/valid.de-en.en"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="exp_runs/raml_work_dir_sample5"
proposal_model_path="work_dir/model.bin"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python -u nmt.py \
    train-mcmc-raml \
    --cuda \
    --vocab ${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --proposal-model ${proposal_model_path} \
    --save-to ${work_dir}/model.bin \
    --input-feed \
    --valid-niter 6700 \
    --batch-size 20 \
    --sample-size 5 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --lr-decay 0.5 2>${work_dir}/err.log

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt
