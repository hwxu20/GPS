
export CUDA_VISIBLE_DEVICES=0,1,2
python run_all_eval.py \
    --test_split ./config/test.list \
    --model_name_or_path ./pretrained_model/T0 \
    --parallelize \
    --template_dir ./ga_t0_t5_lm/ga_configs/result_9 \
    --output_dir ./result_T0_ga_t5_lm
