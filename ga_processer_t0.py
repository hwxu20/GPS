import os
import json
from collections import defaultdict
from tqdm import tqdm
from templates import DatasetTemplates
import yaml
from templates import Template
import uuid
import copy
import torch

os.environ['MKL_THREADING_LAYER'] = 'GNU'
# EN_TASK = ['hellaswag', 'super_glue/copa', 'anli/r1', 'anli/r2', 'anli/r3', 'super_glue/rte',
#            'super_glue/cb', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']

EN_TASK = ['super_glue/wic', 'winogrande/winogrande_xl']


TOP_K_for_each_task = {
    'hellaswag': 4,
    'super_glue/copa': 12,
    'anli/r1': 15,
    'anli/r2': 15,
    'anli/r3': 15,
    'super_glue/rte': 10,
    'super_glue/cb': 15,
    'super_glue/wsc.fixed': 10,
    'super_glue/wic': 10,
    'winogrande/winogrande_xl': 5
}


def _get_pattern(file_name):
    subfile_name = file_name.split('/')[-1][:-5]
    pattern = ''
    for i in range(len(subfile_name)):
        ch = subfile_name[len(subfile_name) - i - 1]
        if ch == '_':
            break
        pattern = ch + pattern
    return int(pattern)


def _dump_json(path, data, beautiful=False):
    with open(path, 'w') as f:
        for i in list(data):
            if beautiful:
                # f.write(json.dumps(i,ensure_ascii=False,sort_keys=True, indent=4) + '\n')
                f.write(json.dumps(i, ensure_ascii=False, indent=4) + '\n')
            else:
                f.write(json.dumps(i, ensure_ascii=False) + '\n')
    print("{} success dumped!".format(path))


def _dump_template(path, template):
    """保存templates"""
    yaml.dump(template, open(path, "w"))


def read_task_template(task_config_dir, task_names):
    """读取template文件"""
    task_configs = {}
    for task_name in task_names:
        name_tuple = task_name.split('/')
        if len(name_tuple) == 2:
            dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
        else:
            dataset_name, dataset_config_name = name_tuple[0], None

        # dataset_templates = DatasetTemplates(f"{dataset_name}" if dataset_config_name is None else f"{dataset_name}/{dataset_config_name}",
        #                                      template_dir=task_config_dir)
        config_file_path = os.path.join(task_config_dir, dataset_name)
        if dataset_config_name:
            config_file_path = os.path.join(config_file_path, dataset_config_name)
        config_file_path = os.path.join(config_file_path, 'templates.yaml')

        # {
        #   dataset: hellaswag
        #   templates_test: Template class
        # }
        task_configs[task_name] = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)

    # print(f'debug: {task_configs}')
    return task_configs


def augment_prompt(input_dir, output_dir, device='0,1', aug_target_num=30, mode='t0'):

    # only chinese dino
    if mode == 'general' or mode == 'cross_task_type':
        return os.system(f'CUDA_VISIBLE_DEVICES={device} python -m torch.distributed.launch --nproc_per_node 2 /mfs/shaonan/moonshot/CPM-1-Generate-main/use_cpm_to_generate.py \
                            --model-parallel-size 2 \
                            --num-layers 32 \
                            --hidden-size 2560 \
                            --load /mfs/shaonan/pretrained_model/CPM-large \
                            --num-attention-heads 32 \
                            --seq-length 1024 \
                            --max-position-embeddings 1024 \
                            --fp16 \
                            --cache-dir cache \
                            --out-seq-length 512 \
                            --temperature 1 \
                            --top_k 0 \
                            --top_p 0.8 \
                            --tokenizer-path /mfs/shaonan/moonshot/CPM-1-Generate-main/bpe_3w_new/ \
                            --vocab-size 30000 \
                            --Top_N_patterns 6 \
                            --input_dir {input_dir} \
                            --output_dir {output_dir} ')
    elif mode == 't0':
        return os.system(f'CUDA_VISIBLE_DEVICES={device} python /home/yanan/shaonan/dino-main/use_dino_to_generate_template_t5.py \
                                    --model_name /home/yanan/shaonan/pretrained_model/t5-xxl-lm-adapt \
                                    --input_dir {input_dir} \
                                    --task_list_file /home/yanan/shaonan/t-zero/config/setting_5/test_temp.list \
                                    --output_dir {output_dir} ')
    else:
        raise NotImplementedError


def run_test(config_dir, exp_dir, device='0,1'):
    # cache_config_dir = '/mfs/yanggang/workspace/megabart_latest/megabart/ga_test_v1/ga_configs/cache'
    # cache_eval_dir = '/mfs/yanggang/workspace/megabart_latest/megabart/ga_test_v1/ga_evals/cache'
    cache_config_dir = os.path.join(exp_dir, 'ga_configs/cache')
    cache_eval_dir = os.path.join(exp_dir, 'ga_evals/cache')
    os.system(f'rm -rf {cache_config_dir}/*')
    os.system(f'cp -r {config_dir}/* {cache_config_dir}')
    os.system(   # TODO: DEBUG
        f'CUDA_VISIBLE_DEVICES={device} python /home/yanan/shaonan/t-zero/evaluation/run_all_eval_seed_val.py \
            --test_split /home/yanan/shaonan/t-zero/config/setting_5/test.list \
            --model_name_or_path /home/yanan/shaonan/pretrained_model/T0 \
            --template_dir {cache_config_dir} \
            --dataset_type ga \
            --ga_val_num 64 \
            --ga_dev_distribution ratio \
            --parallelize \
            --output_dir {cache_eval_dir} ')
    # os.system(f'cp -rf {cache_eval_dir}/* {out_dir}')


##当augment函数在某些task出错或者已经无法继续augment时，为了整体流程的运行，将上一代filter-prompt搬运作为本代结果。
def check_configs(exp_dir, step, mode='general'):
    TASKS = EN_TASK
    if step == 0:
        pass
    else:
        config_dir = os.path.join(exp_dir, 'ga_configs', f'step_{step}')
        config_filter_dir = os.path.join(exp_dir, 'ga_configs_filter', f'step_{step - 1}')
        # current_configs = read_task_configs(config_dir, TASKS)
        current_configs = read_task_template(config_dir, TASKS)
        # current_configs_filter = read_task_configs(config_filter_dir, TASKS)
        current_configs_filter = read_task_template(config_filter_dir, TASKS)
        for task in TASKS:
            if len(current_configs[task]['templates']) == 0:  # TODO: 这里templates_test估计不对
                for cur_uuid, cur_template in current_configs_filter[task]['templates'].items():
                    current_configs[task]['templates'][cur_template.get_id()] = cur_template
                # _dump_json(os.path.join(config_dir, f'{task}.json'), [current_configs[task]], beautiful=True)

                name_tuple = task.split('/')
                if len(name_tuple) == 2:
                    dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
                else:
                    dataset_name, dataset_config_name = name_tuple[0], None
                output_path = os.path.join(config_dir, dataset_name)
                if dataset_config_name:
                    output_path = os.path.join(output_path, dataset_config_name)
                output_path = os.path.join(output_path, 'templates.yaml')

                _dump_template(output_path, current_configs[task])


def ga_process_new(exp_dir, max_steps=1, top_K=6, device='0,1', mode='general'):
    if mode == 't0':
        TASKS = EN_TASK
    else:
        raise NotADirectoryError

    config_dir = os.path.join(exp_dir, 'ga_configs')
    config_filter_dir = os.path.join(exp_dir, 'ga_configs_filter')
    out_dir = os.path.join(exp_dir, 'ga_evals')

    dev_result = defaultdict(dict)
    select_prompt = defaultdict(dict)
    total_configs = []

    # cache_eval_dir = '/mfs/yanggang/workspace/megabart_latest/megabart/ga_test_v1/ga_evals/cache'
    cache_eval_dir = os.path.join(exp_dir, 'ga_evals/cache')

    for ga_step in range(max_steps):
        print(f'processing ... ga_step:{ga_step}')
        # 存放用于当前ga_step的configs
        check_configs(exp_dir, ga_step)   # 如果出错了这里会把上一步的pattern搬过来
        current_config_dir = os.path.join(config_dir, f'step_{ga_step}')   # ga_config/step_x
        # current_configs = read_task_configs(current_config_dir, TASKS)  # 读取当前step下所有config文件
        current_configs = read_task_template(current_config_dir, TASKS)
        total_configs.append(current_configs)    # 每一步的全部config
        # 存放当前ga_step的过滤后的configs，用于下一代的生成
        current_config_filter_dir = os.path.join(config_filter_dir, f'step_{ga_step}')
        if not os.path.exists(current_config_filter_dir):
            os.mkdir(current_config_filter_dir)
        # 存放当前ga_step的dev结果
        current_eval_file = os.path.join(out_dir, f'dev_summary_step_{ga_step}.txt')
        # cache_eval_file = os.path.join(f'{cache_eval_dir}', 'dev_summary.txt')
        # 先把当前要用的config_dir 覆盖 test脚本里的cache 位置
        # 缓存结果不存在时run_test
        if os.path.exists(current_eval_file):
            pass
        # elif ga_step == 0 and os.path.exists(cache_eval_file):
        #     pass
        else:
            # current_config_dir: ga_vx_shot_norm/ga_config, exp_dir: ga_vx_shot_norm
            run_test(current_config_dir, exp_dir, device)
        print(f'ga_step:{ga_step} eval runs to the end ')

        for task in TASKS:
            dev_result[task][f'step_{ga_step}'] = {}
            select_prompt[task][f'step_{ga_step}'] = []
        # 读取之前eval结果
        if not os.path.exists(current_eval_file):
            # 把各个任务的结果合并
            result_file_list = os.listdir(cache_eval_dir)
            result_file_list = [file_name for file_name in result_file_list if file_name.endswith('json')]

            all_result_list = []
            for file_name in result_file_list:
                all_result_list.extend(json.load(open(os.path.join(cache_eval_dir, file_name), 'r')))

            json.dump(all_result_list, open(current_eval_file, 'w'), ensure_ascii=False, indent=4)

        print(f'ga_step:read {ga_step} eval results')
        all_result = json.load(open(current_eval_file, 'r'))
        for result_dict in all_result:
            # print(f'result_dict: {result_dict}')
            # print(f'dev_result: {dev_result}')
            task_name = f'{result_dict["dataset_name"]}/{result_dict["dataset_config_name"]}' \
                if result_dict["dataset_config_name"] else result_dict["dataset_name"]
            dev_result[task_name][f'step_{ga_step}'][result_dict['template_id']] = float(
                result_dict['evaluation']['accuracy'])

        print(f'ga_step:{ga_step} eval statics readed ')
        # print(f'Debug: dev_result: {dev_result}')

        # 选当前step的topk 的prompt
        for task in TASKS:
            # 晒圈当前step的topk
            top_K = TOP_K_for_each_task[task]
            print(f'Select top {top_K} template for task {task}')

            filter_task_config = copy.deepcopy(current_configs[task])
            # 清空templates
            filter_task_config['templates'] = {}

            task_obj = dev_result[task][f'step_{ga_step}']   # 当前任务当前step每个pattern的结果
            # augment template 的id列表,需要保证每个template的uuid是唯一的
            print(f'debug: task_obj for {task} as step {ga_step}: {task_obj}')
            print(f'debug: task_obj.keys(): {task_obj.keys()}')
            print(f'debug: current_configs: {current_configs[task]["templates"]}')
            print(f'debug: current_configs[task][templates].keys(): {current_configs[task]["templates"].keys()}')
            aug_pattern_list = [template_id for template_id in task_obj.keys() if
                                template_id in current_configs[task]['templates'].keys()]
            print(f'debug: aug_pattern_list: {aug_pattern_list}')
            aug_pattern_list.sort(key=lambda x: task_obj.get(x), reverse=True)   # 根据得分对pattern的编号进行排序
            select_prompt[task][f'step_{ga_step}'] = aug_pattern_list[:top_K]   # select_prompt是每个task每一步取topk的结果
            for origin_pattern in aug_pattern_list[:top_K]:
                # 当前step的config
                filter_task_config['templates'][origin_pattern] = current_configs[task]['templates'][origin_pattern]

            # current_config_filter_file = os.path.join(current_config_filter_dir, f'{task}.json')
            name_tuple = task.split('/')
            if len(name_tuple) == 2:
                dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
            else:
                dataset_name, dataset_config_name = name_tuple[0], None
            current_config_filter_file = os.path.join(current_config_filter_dir, dataset_name)
            if dataset_config_name:
                current_config_filter_file = os.path.join(current_config_filter_file, dataset_config_name)
            os.makedirs(current_config_filter_file, exist_ok=True)
            current_config_filter_file = os.path.join(current_config_filter_file, 'templates.yaml')

            if not os.path.exists(current_config_filter_file):
                # 保存当前任务的config
                # _dump_json(current_config_filter_file, [filter_task_config], beautiful=True)
                _dump_template(current_config_filter_file, filter_task_config)
            # 并对于每个任务进行排序，取topK个进行保存（不需要重新编号），
        # 如果ga_step小于max_steps-1 则对保存的TopK个config进行再次的augment
        next_configs_dir = os.path.join(config_dir, f'step_{ga_step + 1}')
        if ga_step < max_steps - 1 and not os.path.exists(next_configs_dir):
            stat_code = augment_prompt(current_config_filter_dir, next_configs_dir, device=device, mode=mode)
            if stat_code != 0:
                max_steps = ga_step + 1
                break
        print(f'ga_step:{ga_step} filter and augment has done ')

    print('starting merge result configs...')
    result_dir = os.path.join(config_dir, f'result_{max_steps}')
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    ga_result_log = open(os.path.join(result_dir, 'ga_result_log.txt'), 'w')
    ga_result_log.write('\t'.join(['task', 'work?', 'base_best_score', 'ga_prompts_score']) + '\n')
    # 全部ga_step完成后，根据除了step0的所有遗传模版取top_5保存下来作为ga_test_prompts，并判断在各任务上是否能够大于step0的best prompt
    for task in TASKS:
        top_K = TOP_K_for_each_task[task]
        select_prompt_list = []  # id 列表
        select_metric_list = []
        max_base_metric = 0
        template_set = set()
        ga_prompts_scores = []
        prompt_count = 0
        # 遍历每个step的每个选出的pattern
        for ga_step in range(max_steps):
            for p in select_prompt[task][f'step_{ga_step}']:
                select_prompt_list.append(f'{ga_step}_{p}')  # eg. 第3步第7个step: 3_7
                m = dev_result[task][f'step_{ga_step}'][p]   # 这个prompt的效果
                select_metric_list.append(m)
                if ga_step == 0 and m > max_base_metric:   # 记录原始prompt里最好的效果
                    max_base_metric = m
        index_select = list(range(len(select_metric_list)))  # [0, 1, 2 ... 所有选出的p的数量]
        index_select.sort(key=lambda x: select_metric_list[x], reverse=True)   # 所有选出的p的效果排序
        # result = [select_prompt_list[i] for i in index_select[:top_K]]
        result = [select_prompt_list[i] for i in index_select]  # [1_3, 4_7, ...]

        result_task_config = copy.deepcopy(total_configs[0][task])
        # 清空templates
        result_task_config['templates'] = {}

        for id, p_obj in enumerate(result):   # {step}_{template_id}
            if prompt_count >= top_K:
                break
            step, template_id = p_obj.split('_')
            template_obj = total_configs[int(step)][task]['templates'][template_id]
            # template_obj = ''.join(template_obj)
            if template_obj not in template_set:
                result_task_config['templates'][template_obj.get_id()] = total_configs[int(step)][task]['templates'][template_id]
                template_set.add(template_obj)
                ga_prompts_scores.append(index_select[id])
                prompt_count += 1
        work = (result[0].split('_')[0] != '0')
        ga_prompts_score = ','.join([f'{select_metric_list[i]:.4f}' for i in ga_prompts_scores[:top_K]])
        ga_result_log.write('\t'.join([task, str(work), f'{max_base_metric:.4f}', ga_prompts_score]) + '\n')

        name_tuple = task.split('/')
        if len(name_tuple) == 2:
            dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
        else:
            dataset_name, dataset_config_name = name_tuple[0], None
        result_file_name = os.path.join(result_dir, dataset_name)
        if dataset_config_name:
            result_file_name = os.path.join(result_file_name, dataset_config_name)
        os.makedirs(result_file_name, exist_ok=True)
        result_file_name = os.path.join(result_file_name, 'templates.yaml')
        _dump_template(result_file_name, result_task_config)

    ga_result_log.close()

    print('ga process done')


def step_0_prompts(step_0_dir, mode='spcific'):

    # mode opts:[spcific,random,LM_BFF]
    # spcific_dir = '/mfs/shaonan/moonshot/megabart_yg/tasks/task_configs_specific'
    spcific_dir = '/home/yanan/shaonan/t-zero/templates_test'

    os.system(f'cp -rf {spcific_dir}/* {step_0_dir}')


if __name__ == '__main__':
    # [run_pipline ,cache,baseline,-]
    mode = 'run_pipline'
    # [portrait,general]
    # task_mode = 'general'
    task_mode = 't0'
    # task_mode = 'cross_task_type'

    device = '1,2,3'
    if mode == 'run_pipline':
        # 把卡占了，别让别人中途提任务
        # fake_model_1 = torch.nn.Linear(1000, 1000)
        # fake_model_1 = fake_model_1.cuda(1)
        # fake_model_2 = torch.nn.Linear(1000, 1000)
        # fake_model_2 = fake_model_2.cuda(2)
        # fake_model_3 = torch.nn.Linear(1000, 1000)
        # fake_model_3 = fake_model_3.cuda(2)

        # 除了这个dir，还需要更新run_test的脚本、以及augment的方式
        exp_dir = '/home/yanan/shaonan/t-zero/evaluation/ga_t0_val64_seed43_t5_lm'
        if not os.path.exists(os.path.join(exp_dir, 'ga_configs')):
            os.makedirs(os.path.join(exp_dir, 'ga_configs'), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, 'ga_configs', 'cache'), exist_ok=True)
            step_0_dir = os.path.join(exp_dir, 'ga_configs', 'step_0')
            os.makedirs(step_0_dir)
            step_0_prompts(step_0_dir, mode='spcific')
            os.mkdir(os.path.join(exp_dir, 'ga_configs_filter'))
            os.mkdir(os.path.join(exp_dir, 'ga_evals'))
            os.mkdir(os.path.join(exp_dir, 'tensorboard_unified'))

        ga_process_new(exp_dir=exp_dir, max_steps=7, top_K=12, device=device, mode=task_mode)
        # print(f'model_1: {fake_model_1}, model_2: {fake_model_2}, model_3: {fake_model_3}')
    elif mode == 'cache':
        logs_dir = '/mfs/shaonan/moonshot/megabart_yg/ga_cross_task_type_v6_norm_shot'
        # get_Baseline_by_filterLog(logs_dir, max_steps=6, mode='portrait')
    elif mode == 'baseline':
        device = '1,9'

        print(f'ga_step:baseline eval statics has been generated ')

    else:
        # result_dir = '/mfs/yanggang/workspace/megabart_latest/megabart/ga_portrait_v7_shot_norm/result_step_6'
        # print(len(PORTRAIT_TASK))
        # portrait_config_factory(exp_dir='/mfs/yanggang/workspace/megabart_latest/megabart/ga_test_v6/',step=0)
        # portrait_excel_factory(result_dir,step=7,mode ='result')
        device = '1,9'
        # portrait_config_factory(
        #     exp_dir='/mfs/yanggang/workspace/megabart_latest/megabart/ga_portrait_v7_shot_norm_old/ga_configs_filter/step_0_c',
        #     step=0, mode='result')
        augment_prompt(
            input_dir='/mfs/yanggang/workspace/megabart_latest/megabart/ga_portrait_v7_shot_norm_old/ga_configs_filter/step_0_c',
            output_dir='/mfs/yanggang/workspace/megabart_latest/megabart/ga_portrait_v7_shot_norm_old/ga_configs_filter/step_1_c',
            mode='portrait', device=device)
