CUDA_VISIBLE_DEVICES=0,1,2 python use_dino_to_generate_template_t5.py  \
                        --task_list_file ../config/test.list \
                        --model_name ../pretrained_model/t5-xxl-lm-adapt \
                        --input_dir ../templates \
                        --output_dir ./temp_dir \

