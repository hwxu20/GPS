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
EN_TASK = ['hellaswag', 'super_glue/copa', 'anli/r1', 'anli/r2', 'anli/r3', 'super_glue/rte',
           'super_glue/cb', 'super_glue/wsc.fixed', 'super_glue/wic', 'winogrande/winogrande_xl']


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
                f.write(json.dumps(i, ensure_ascii=False, indent=4) + '\n')
            else:
                f.write(json.dumps(i, ensure_ascii=False) + '\n')
    print("{} success dumped!".format(path))


def _dump_template(path, template):
    yaml.dump(template, open(path, "w"))


def read_task_template(task_config_dir, task_names):
    """read template file for each task"""
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

    return task_configs


def augment_prompt(input_dir, output_dir, device='0,1', mode='t0'):
    """generate new prompts for each task"""
    if mode == 't0':
        return os.system(f'CUDA_VISIBLE_DEVICES={device} python ./dino/use_dino_to_generate_template_t5.py \
                                    --model_name ./pretrained_model/t5-xxl-lm-adapt \
                                    --input_dir {input_dir} \
                                    --task_list_file ./config/test.list \
                                    --output_dir {output_dir} ')
    else:
        raise NotImplementedError


def run_test(config_dir, exp_dir, device='0,1'):
    cache_config_dir = os.path.join(exp_dir, 'ga_configs/cache')
    cache_eval_dir = os.path.join(exp_dir, 'ga_evals/cache')
    os.system(f'rm -rf {cache_config_dir}/*')
    os.system(f'cp -r {config_dir}/* {cache_config_dir}')
    os.system(   # TODO: DEBUG
        f'CUDA_VISIBLE_DEVICES={device} python ./run_all_eval.py \
            --test_split ./config/test.list \
            --model_name_or_path ./pretrained_model/T0 \
            --template_dir {cache_config_dir} \
            --dataset_type ga \
            --ga_dev_distribution ratio \
            --parallelize \
            --output_dir {cache_eval_dir} ')


def check_configs(exp_dir, step):
    """Check whether the algorithm raises errors"""
    TASKS = EN_TASK
    if step == 0:
        pass
    else:
        config_dir = os.path.join(exp_dir, 'ga_configs', f'step_{step}')
        config_filter_dir = os.path.join(exp_dir, 'ga_configs_filter', f'step_{step - 1}')
        current_configs = read_task_template(config_dir, TASKS)
        current_configs_filter = read_task_template(config_filter_dir, TASKS)
        for task in TASKS:
            if len(current_configs[task]['templates']) == 0:
                for cur_uuid, cur_template in current_configs_filter[task]['templates'].items():
                    current_configs[task]['templates'][cur_template.get_id()] = cur_template

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


def ga_process(exp_dir, max_steps=1, device='0,1', mode='t0'):
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

    cache_eval_dir = os.path.join(exp_dir, 'ga_evals/cache')

    for ga_step in range(max_steps):
        print(f'processing ... ga_step:{ga_step}')

        check_configs(exp_dir, ga_step)   # heck whether the algorithm raises errors in the previous step
        current_config_dir = os.path.join(config_dir, f'step_{ga_step}')   # ga_config/step_x
        current_configs = read_task_template(current_config_dir, TASKS)
        total_configs.append(current_configs)    # template for each task in each step
        current_config_filter_dir = os.path.join(config_filter_dir, f'step_{ga_step}')

        if not os.path.exists(current_config_filter_dir):
            os.mkdir(current_config_filter_dir)
        # eval results for each task
        current_eval_file = os.path.join(out_dir, f'dev_summary_step_{ga_step}.txt')

        if os.path.exists(current_eval_file):
            pass
        else:
            # current_config_dir: ga_vx_shot_norm/ga_config, exp_dir: ga_vx_shot_norm
            run_test(current_config_dir, exp_dir, device)
        print(f'ga_step:{ga_step} eval runs to the end ')

        for task in TASKS:
            dev_result[task][f'step_{ga_step}'] = {}
            select_prompt[task][f'step_{ga_step}'] = []
        if not os.path.exists(current_eval_file):
            result_file_list = os.listdir(cache_eval_dir)
            result_file_list = [file_name for file_name in result_file_list if file_name.endswith('json')]

            all_result_list = []
            for file_name in result_file_list:
                all_result_list.extend(json.load(open(os.path.join(cache_eval_dir, file_name), 'r')))

            json.dump(all_result_list, open(current_eval_file, 'w'), ensure_ascii=False, indent=4)

        print(f'ga_step:read {ga_step} eval results')
        all_result = json.load(open(current_eval_file, 'r'))
        for result_dict in all_result:
            task_name = f'{result_dict["dataset_name"]}/{result_dict["dataset_config_name"]}' \
                if result_dict["dataset_config_name"] else result_dict["dataset_name"]
            dev_result[task_name][f'step_{ga_step}'][result_dict['template_id']] = float(
                result_dict['evaluation']['accuracy'])

        print(f'ga_step:{ga_step}, read eval statics done.')

        # select topk prompt
        for task in TASKS:
            # note: k is different for different tasks
            top_K = TOP_K_for_each_task[task]
            print(f'Select top {top_K} template for task {task}')

            filter_task_config = copy.deepcopy(current_configs[task])
            filter_task_config['templates'] = {}

            task_obj = dev_result[task][f'step_{ga_step}']
            print(f'debug: task_obj for {task} as step {ga_step}: {task_obj}')
            print(f'debug: task_obj.keys(): {task_obj.keys()}')
            print(f'debug: current_configs: {current_configs[task]["templates"]}')
            print(f'debug: current_configs[task][templates].keys(): {current_configs[task]["templates"].keys()}')
            aug_pattern_list = [template_id for template_id in task_obj.keys() if
                                template_id in current_configs[task]['templates'].keys()]
            print(f'debug: aug_pattern_list: {aug_pattern_list}')
            aug_pattern_list.sort(key=lambda x: task_obj.get(x), reverse=True)
            select_prompt[task][f'step_{ga_step}'] = aug_pattern_list[:top_K]
            for origin_pattern in aug_pattern_list[:top_K]:
                filter_task_config['templates'][origin_pattern] = current_configs[task]['templates'][origin_pattern]

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
                # dump selected template
                _dump_template(current_config_filter_file, filter_task_config)
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
    for task in TASKS:
        top_K = TOP_K_for_each_task[task]
        select_prompt_list = []
        select_metric_list = []
        max_base_metric = 0
        template_set = set()
        ga_prompts_scores = []
        prompt_count = 0

        for ga_step in range(max_steps):
            for p in select_prompt[task][f'step_{ga_step}']:
                select_prompt_list.append(f'{ga_step}_{p}')
                m = dev_result[task][f'step_{ga_step}'][p]
                select_metric_list.append(m)
                if ga_step == 0 and m > max_base_metric:
                    max_base_metric = m
        index_select = list(range(len(select_metric_list)))
        index_select.sort(key=lambda x: select_metric_list[x], reverse=True)
        result = [select_prompt_list[i] for i in index_select]

        result_task_config = copy.deepcopy(total_configs[0][task])
        result_task_config['templates'] = {}

        for id, p_obj in enumerate(result):   # {step}_{template_id}
            if prompt_count >= top_K:
                break
            step, template_id = p_obj.split('_')
            template_obj = total_configs[int(step)][task]['templates'][template_id]
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


def step_0_prompts(step_0_dir):
    spcific_dir = './templates'

    os.system(f'cp -rf {spcific_dir}/* {step_0_dir}')


if __name__ == '__main__':
    mode = 'run_pipline'
    task_mode = 't0'

    device = '0,1,2'
    exp_dir = 'ga_t0_t5_lm_maxstep9'
    if not os.path.exists(os.path.join(exp_dir, 'ga_configs')):
        os.makedirs(os.path.join(exp_dir, 'ga_configs'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'ga_configs', 'cache'), exist_ok=True)
        step_0_dir = os.path.join(exp_dir, 'ga_configs', 'step_0')
        os.makedirs(step_0_dir)
        step_0_prompts(step_0_dir)
        os.mkdir(os.path.join(exp_dir, 'ga_configs_filter'))
        os.mkdir(os.path.join(exp_dir, 'ga_evals'))
        os.mkdir(os.path.join(exp_dir, 'tensorboard_unified'))

    ga_process(exp_dir=exp_dir, max_steps=9, device=device, mode=task_mode)
