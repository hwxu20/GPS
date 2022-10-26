CUDA_VISIBLE_DEVICES=0,1,2 python use_dino_to_generate_template_t5.py  \
                        --task_list_file /home/yanan/shaonan/t-zero/config/setting_5/test_temp.list \
                        --model_name /home/yanan/shaonan/pretrained_model/t5-xxl-lm-adapt \
                        --input_dir /home/yanan/shaonan/t-zero/templates \
                        --output_dir /home/yanan/shaonan/data/temp_dir \

                        # --input_dir /mfs/shaonan/moonshot/t-zero/templates
                        # --model_name /mfs/shaonan/pretrained_model/t5-xxl-lm-adapt
