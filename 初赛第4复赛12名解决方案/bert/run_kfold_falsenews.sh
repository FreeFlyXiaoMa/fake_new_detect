python data_split.py 13
mkdir ./ckpt_base13
for IDX in $(seq 1 13) 
do
    mkdir ./ckpt_base13/ckpt_${IDX}
    python -u run_classifier.py \
                   --train_set train_${IDX}.csv \
                   --dev_set dev_${IDX}.csv \
                   --test_save ./test/base13/result_${IDX}.csv \
                   --output_dir ./ckpt_base13/ckpt_${IDX} 
done
rm ./task1/dev_*
rm ./task1/train_*

