python data_split.py 15

mkdir ./ckpt_base15test

for IDX in $(seq 1 15)
do
    mkdir ./ckpt_base15test/ckpt_${IDX}
    python -u run_classifier_xlnet.py \
                   --train_set train_${IDX}.csv \
                   --dev_set dev_${IDX}.csv \
                   --test_save ./test/base15/result_${IDX}.csv \
                   --model_dir ./ckpt_base15test/ckpt_${IDX} \
                   --output_dir ./ckpt_base15test/ckpt_${IDX} 
done
rm ./task1/dev_*
rm ./task1/train_*

