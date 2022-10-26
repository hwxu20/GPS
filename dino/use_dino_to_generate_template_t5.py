import argparse
import copy
import json
import os
import re
import json
import string
from datetime import datetime

from modeling import GPT2Wrapper, DinoGenerator
from utils import set_seed

from templates import Template
import yaml
import uuid


def get_valid_invalid_option(option_list):
    """返回不合法的option单词，加入黑名单中"""
    option_list = [option.lower().strip() for option in option_list]
    black_set = set()
    if option_list.count('yes') > 0:  # yes/no/(may)
        black_set.add('false')
    elif option_list.count('correct') > 0:  # correct/incorrect/inconlusive
        black_set.add('false')
    elif option_list.count('false') > 0:  # True/false
        pass

    # 不能直接加no，不然find函数会把带no的单词都算命中

    return black_set


def will_keep_with_clean_rule(task_name, template_name, description, option_list):
    """判断生成的description是否合法"""

    description = description.strip()
    description = description.replace('  ', ' ')

    # 长度控制
    if len(description) < 2:
        return False
    if len(description.split(' ')) < 2:
        return False

    if description.count('{') != description.count('}'):
        return False

    MUST_WORD = set()
    BLACK_WORD = {'=', '#', '__', '[', ']', '\\u', '%', '{{{', '}}}', 'God', '\\x', 'ie.', 'i.e.',
                  '(2)', '(1)', '(?', 'ENSLAVED', 'DevOps', 'U.S.', '**', '...', '+', '--', '(a)',
                  '(1)'}
    # 黑名单，过滤有特殊符号的prompt

    VALID_PLACEHOLDER = {'hypothesis', 'premise', 'ctx', 'pronoun', 'reference', 'word', 'option' 'option1', 'option2'}

    # 必须是合法placeholder
    re_pattern = re.compile(r'{{(.*?)}}', re.S)  # 最小匹配
    match_placeholder = re.findall(re_pattern, description)  # 必须是合法的这几个placeholder
    for placeholder in match_placeholder:
        # 每个placeholder的个数不能超过1
        if match_placeholder.count(placeholder) > 1:
            return False

        placeholder = placeholder.strip()
        if placeholder not in VALID_PLACEHOLDER:
            return False
    if len(match_placeholder) > 2:
        return False

    if template_name in ['should assume', 'does it follow that',
                         'must be true', 'can we infer', 'guaranteed true']:
        if description.find('premise') == -1 or description.find('hypothesis') == -1:
            return False
        if description.count('premise') != 1 or description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
    elif template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        if description.find('hypothesis') == -1:
            return False
        if description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
        BLACK_WORD.update({'Is there'})
    elif template_name in ['does this imply']:
        if description.find('hypothesis') == -1:
            return False
        if description.count('hypothesis') != 1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
    elif template_name in ['claim true/false/inconclusive']:
        if description.find(':') == -1:
            return False
        if description.find('hypothesis') == -1:
            return False
        BLACK_WORD.update(get_valid_invalid_option(option_list))
        BLACK_WORD.update({'not true'})

    # hellaswag
    elif template_name in ['complete_first_then']:
        MUST_WORD.add('end')
    elif template_name in ['Predict ending with hint']:
        MUST_WORD.add('end')
        BLACK_WORD.update({'What are', 'Whats', 'Why'})
    elif template_name in ['if_begins_how_continues']:
        if description.find('this.') == -1 and description.find('this,') == -1:
            return False
        BLACK_WORD.update({'What', 'different', 'Imagine'})

    # copa
    elif template_name in ['exercise']:
        MUST_WORD.add('alternative')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'least'})
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        MUST_WORD.add('happen')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'want',
                           'lucky', 'possibility', 'odds'})
    elif template_name in ['i_am_hesitating']:
        if not description[:-1].endswith('option'):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject'})
    elif template_name in ['plausible_alternatives']:
        # MUST_WORD.add('option')  # t5生成质量很好，不需要这个
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'less'})
    elif template_name in ['\u2026As a result, C1 or C2?']:
        if not description.endswith(','):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'teaching'})
    elif template_name in ['\u2026which may be caused by']:
        if description[-1] in string.punctuation:
            return False  # 不能以标点符号结尾
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'NOT'})
    elif template_name in ['choose']:
        if not description.endswith(':'):
            return False
    elif template_name in ['more likely']:
        MUST_WORD.add('continuation')
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'less', 'least'})
    elif template_name in ['cause_effect']:
        if not description[:-1].endswith('option'):
            return False
        BLACK_WORD.update({'three', 'four', 'five', 'six', 'How many', 'subject', 'Do you'})

    # wsc.fixed
    elif template_name in ['does the pronoun refer to', 'does p stand for']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['replaced with', 'the pronoun refers to']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None  # 只适用于wsc
        if description.find('reference') == -1 or description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
    elif template_name in ['by p they mean']:
        pass

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which'})
    elif template_name in ['grammar_homework']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'How'})
    elif template_name in ['affirmation_true_or_false']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'What'})
    elif template_name in ['same_sense']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['question-context']:
        if description.find('word') == -1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['polysemous']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})
    elif template_name in ['similar-sense']:
        if description.find('word') == -1:
            return False
        if description.count('word') != 1:
            return False
        BLACK_WORD.update({'friend', 'watch', 'verb', 'different', 'die', 'which', 'American'})

    # winogrande
    elif template_name in ['does underscore refer to']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        if description[-1] in string.punctuation:
            return False  # 不能以标点符号结尾
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['stand for']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
        if not description[:-1].endswith('reference'):
            return False
        BLACK_WORD.update({'is', 'which', '\'', 'Why'})
    elif template_name in ['underscore refer to']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        BLACK_WORD.update({'If', 'subject', 'object', 'function', '\'',  'Why'})
    elif template_name in ['fill in the blank']:
        if description.find('blank') == -1:
            return False
        if description.count('blank') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['True or False']:
        if description.find('pronoun') == -1 or description.find('reference') == -1:
            return False
        if description.count('pronoun') != 1 or description.count('reference') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    elif template_name in ['Replace']:
        if description.find('pronoun') == -1:
            return False
        if description.count('pronoun') != 1:
            return False
        BLACK_WORD.update({'\'', 'Why'})
    else:
        pass

    # 必须包含这些词
    for word in MUST_WORD:
        if description.find(word) == -1:
            return False

    # 不能包含黑名单里的词
    for word in BLACK_WORD:
        if description.find(word) != -1:
            return False

    return True


def get_description_from_template(task_name, template):
    """从template obj中提取description，需要为每个template编写规则, 对于不使用的template返回None"""
    if not template.metadata.original_task:
        return None

    template_text = template.jinja
    template_name = template.name

    description = None

    if template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', '')
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['should assume', 'must be true', 'can we infer',
                           'guaranteed true']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', 'premise')
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['does it follow that']:
        description = template_text.split('|||')[0]
        description = description.replace('{{premise}}', 'premise')
        description = description.replace('{{hypothesis}}', 'hypothesis')
    elif template_name in ['does this imply']:
        description = template_text.split('|||')[0]
        description = description.split('\n\n')[1]
        description = description.replace('"{{hypothesis}}"', 'hypothesis')
    elif template_name in ['claim true/false/inconclusive']:
        description = template_text.split(':')[0]
        description = description.replace('{{premise}}', '')
        description = description.strip()
        description += ': hypothesis true or false?'
    # elif template_name in ['GPT-3 style']:
    #     pass  # 无法augment, 没有prompt
    elif template_name in ['take the following as truth', 'guaranteed/possible/impossible',
                           'always/sometimes/never',
                           'consider always/sometimes/never']:
        pass  # 无法augment, 不好处理

    # hellaswage
    elif template_name in ['complete_first_then']:
        description = template_text.split(':')[0]
    elif template_name in ['Randomized prompts template']:
        pass  # 无法augment, 不好处理
    elif template_name in ['Predict ending with hint']:
        description = template_text.split('\n')[0]
    elif template_name in ['if_begins_how_continues']:
        description = template_text.split('\n\n')[0]
        description = description.replace(': {{ ctx }}...', '.')

    # copa
    elif template_name in ['exercise']:
        description = template_text.split('\n')[0]
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        description = template_text.split('{{ premise }}')[1]
        description = description.split('"{{ answer_choices[0] }}')[0]
    elif template_name in ['i_am_hesitating']:
        description = template_text.split('\n\n')[1]
        description = description.split('{% if question == "cause" %}')[0]
        description += ' option.'
    elif template_name in ['plausible_alternatives']:
        description = template_text.split('\n')[1]
    elif template_name in ['C1 or C2? premise, so/because\u2026', "\u2026why? C1 or C2"]:
        pass  # 无法augment, 没有prompt
    elif template_name in ['choose']:
        description = template_text.split('\n')[1]
    elif template_name in ['\u2026As a result, C1 or C2?', '\u2026which may be caused by']:
        description = template_text.split('{{ premise }}')[1]
        description = description.split('"{{ answer_choices[0] }}"')[0]
    elif template_name in ['best_option']:
        pass  # 无法augment, 不好处理
    elif template_name in ['more likely']:
        description = template_text.split('\n')[0]
    elif template_name in ['cause_effect']:
        description = template_text.split('\n\n')[1]
        description = description.split('{% if question == "cause" %}')[0]
        description = description.strip()
        description += ' option.'

    # rte/CB与anli完全相同
    # wsc.fixed
    elif template_name in ['does the pronoun refer to']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text.lower() }}"', '')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['does p stand for']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text.lower() }}"', 'the pronoun')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['replaced with']:
        description = template_text.split('|||')[0]
        description = description.replace('{{ text }} ', '')
        description = description.replace('"{{ span2_text }}"', '')
        description = description.replace('"{{ span1_text }}"', 'the reference')
    elif template_name in ['the pronoun refers to']:
        description = template_text.split('|||')[0]
        description = description.split('\n')[1]
        description = description.replace('"{{ span2_text }}"', '')
        description = description.replace('{{ span1_text }}', 'the reference')
    elif template_name in ['in other words', 'I think they mean', 'p is/are r',
                           'Who or what is/are', 'by p they mean']:
        pass  # 无法处理
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None
        description = template_text.split('\n\n')[1]
        description = description.replace('\"{{ span2_text }}\"', '')
        description = description.replace('{{ span1_text }}', 'the reference')

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        description = template_text.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['grammar_homework']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['affirmation_true_or_false']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', 'The word')
    elif template_name in ['same_sense']:
        description = template_text.split('\n\n')[1]
        description = description.split('\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        description = template_text.split('\n')[2]
        description = description.replace('"{{word}}"', '')
        description = description.replace('\'{{word}}\'', '')
        description = description.replace('{{word}}', '')
    elif template_name in ['question-context']:
        description = template_text.split('\n')[0]
        description = description.replace('\'{{word}}\'', '')
    elif template_name in ['polysemous']:
        description = template_text.split('\n\n')[0]
        description = description.replace('"{{word}}"', '')
    elif template_name in ['similar-sense']:
        description = template_text.split('\n')[2]
        description = description.replace('{{word}}', 'the word')
        description = description.strip()

    # winogrande
    elif template_name in ['does underscore refer to']:
        description = template_text.split('{{ option1 }}')[0]
        description = description.replace('{{ sentence }} ', '')
        description = description.replace('_', 'the pronoun')
    elif template_name in ['stand for']:
        description = template_text.split('{{answer_choices[0]}}')[0]
        description = description.replace('_', 'pronoun')
        description += 'the reference?'
    elif template_name in ['underscore refer to']:
        description = template_text.split('\n')[1]
        description = description.split('{{ option1 }}')[0]
        description = description.strip()
        description = description.replace('_', 'pronoun')
    elif template_name in ['fill in the blank']:
        description = template_text.split('\n')[0]
        description = description.replace('_', 'blank')
    elif template_name in ['True or False']:
        description = template_text.split('\n')[0]
        description = description.replace('_', 'pronoun')
        description = description.replace('{{option1}}', 'the reference')
    elif template_name in ['Replace']:
        description = template_text.split('\n')[1]
        description = description.replace('_', 'pronoun')
    else:
        print(f'无法识别的template: {template_name}')

    # debug
    # if description is not None:
    #     print(f'template_name: {template_name}\ntemplate_text: {template_text}\ndescription: {description}')
    if description is not None:
        description = description.replace('  ', ' ')
        description = description.strip()

    return description


def build_template(task_name: str, template_name, augment_description):
    """根据description和template_name生成template"""
    augment_template = None
    if template_name in ['must be true', 'can we infer', 'guaranteed true']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' "{{hypothesis}}" ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['should assume']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' "{{hypothesis}}" ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does it follow that']:
        augment_template = augment_description.replace('premise', ' {{premise}} ')
        augment_template = augment_template.replace('hypothesis', ' {{hypothesis}} ')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['MNLI crowdsource', 'based on the previous passage', 'justified in saying']:
        augment_template = '{{premise}} ' + augment_description
        augment_template = augment_template.replace('hypothesis', '"{{hypothesis}}"')
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does this imply']:
        augment_template = augment_description.replace('hypothesis', '"{{hypothesis}}"')
        if task_name.startswith('anli'):
            augment_template = '{{premise}} \n\n' + augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = '{{premise}} \n\n' + augment_template + ' ||| {% if label != -1 %}{{answer_choices[label]}}{% endif %}'
    elif template_name in ['claim true/false/inconclusive']:
        augment_template = augment_description.split(':')[0]
        augment_template = '{{premise}} ' + augment_template + \
                           ': "{{hypothesis}}" {{"true"}}, {{"false"}}, or {{"inconclusive"}}?'
        if task_name.startswith('anli'):
            augment_template = augment_template + ' ||| {{ answer_choices[label] }}'
        elif task_name.startswith('super_glue'):
            augment_template = augment_template + ' ||| {% if label !=-1 %}{{ answer_choices[label] }}{% endif %}'

    # hellaswage
    elif template_name in ['complete_first_then']:
        if augment_description[-1] in string.punctuation:
            augment_description = augment_description[:-1]
        augment_template = augment_description + ':\n'
        augment_template = augment_template + 'First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }} ...\n\n(a) {{ answer_choices[0] }}\n\n(b) {{ answer_choices[1] }}\n\n(c) {{ answer_choices[2] }}\n\n(d) {{ answer_choices[3] }}\n|||\n{{ answer_choices[label | int()] }}'
    elif template_name in ['Predict ending with hint']:
        augment_template = augment_description + '\n{{ctx}}\n\n(a)  {{answer_choices[0]}}\n\n(b)  {{answer_choices[1]}}\n\n(c)  {{answer_choices[2]}}\n\n(d)  {{answer_choices[3]}}\n\nHint: the topic of the sentence is {{activity_label}}\n|||\n{{answer_choices [label | int()]}}'
    elif template_name in ['if_begins_how_continues']:
        if augment_description.find('this.') != -1:
            augment_template = augment_description.replace('this.', 'this: {{ ctx }}...')
        else:
            augment_template = augment_description.replace('this,', 'this: {{ ctx }}...')
        augment_template = augment_template + ' \n\nEnding 1: {{ endings[0] }}\n\nEnding 2: {{ endings[1] }}\n\nEnding 3: {{ endings[2] }}\n\nEnding 4: {{ endings[3] }}\n|||{{answer_choices[label | int()] }}'

    # copa
    elif template_name in ['exercise']:
        if not augment_description.startswith('Exercise'):
            augment_description = 'Exercise: ' + augment_description
        augment_template = augment_description + '\n\n{{ premise }} {% if question == "cause" %} because... {% else %} so... {% endif%}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['\u2026What could happen next, C1 or C2?']:
        augment_template = '{% if question == \"effect\" %} \n{{ premise }} ' + augment_description + \
            ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['i_am_hesitating']:
        augment_template = '{{ premise }} \n\n' + augment_description
        augment_template = augment_template[:-(len('option.'))]
        augment_template = augment_template + '{% if question == "cause" %} cause: {% else %} effect: {% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['plausible_alternatives']:
        augment_template = '{{ premise }} {% if question == "cause" %} This happened because... {% else %} As a consequence... {% endif %}\n' + augment_description
        augment_template = augment_template + '\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['\u2026which may be caused by']:
        augment_template = '{% if question == \"cause\" %} \n{{ premise }} ' + augment_description
        augment_template = augment_template + ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['\u2026As a result, C1 or C2?']:
        augment_template = '{% if question == \"effect\" %} \n{{ premise }} ' + augment_description
        augment_template = augment_template + ' \"{{ answer_choices[0] }}\" or \"{{ answer_choices[1] }}\"? ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}\n{% endif %}'
    elif template_name in ['choose']:
        augment_template = '{{ premise }} {% if question == "cause" %} because... {% else %} so... {% endif %}\n' + augment_description
        augment_template = augment_template + '\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['more likely']:
        augment_template = augment_description + '\n{{ premise }} {% if question == "cause" %} as a result of: {% else %} as a consequence:{% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'
    elif template_name in ['cause_effect']:
        augment_template = '{{ premise }}\n\n' + augment_description
        augment_template = augment_template[:-len('option.')]
        augment_template = augment_template + ' {% if question == "cause" %} cause: {% else %} effect:{% endif %}\n- {{choice1}}\n- {{choice2}} ||| {% if label != -1 %}{{ answer_choices[label] }}{%endif%}'

    # wsc.fixed
    elif template_name in ['does the pronoun refer to']:
        augment_template = '{{ text }} ' + augment_description
        augment_template = augment_template.replace('pronoun', ' "{{ span2_text.lower() }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + '||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['does p stand for']:
        augment_template = '{{ text }} ' + augment_description
        if augment_template.find('the pronoun') != -1:
            augment_template = augment_template.replace('the pronoun', ' "{{ span2_text.lower() }}" ')
        else:
            augment_template = augment_template.replace('pronoun', ' "{{ span2_text.lower() }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['replaced with']:
        augment_template = '{{ text }} ' + augment_description
        augment_template = augment_template.replace('pronoun', 'pronoun "{{ span2_text }}" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' "{{ span1_text }}" ')
        else:
            augment_template = augment_template.replace('reference', ' "{{ span1_text }}" ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['the pronoun refers to']:
        augment_template = '{{ text }} \n' + augment_description
        augment_template = augment_template.replace('pronoun', 'pronoun {{ span2_text }} ')
        augment_template = augment_template.replace('{{reference}}', ' {{ span1_text }} ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
        augment_template = augment_template + ' ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
    elif template_name in ['GPT-3 Style', 'GPT-3 style']:
        if task_name.find('wsc') == -1:
            return None
        augment_template = 'Passage: {{ text }} \n\n' + augment_description + '\n\nAnswer: ||| {% if label != -1 %}{{ answer_choices[label] }}{% endif %}'
        augment_template = augment_template.replace('pronoun', 'pronoun \"{{ span2_text }}\" ')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', ' {{ span1_text }} ')
        else:
            augment_template = augment_template.replace('reference', ' {{ span1_text }} ')
    elif template_name in ['by p they mean']:
        pass

    # wic
    elif template_name in ['question-context-meaning-with-label', 'question-context-meaning']:
        augment_description = augment_description.replace('words', 'word')
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['grammar_homework']:
        augment_template = 'Homework\n\n' + augment_description
        augment_template = augment_template.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['affirmation_true_or_false']:
        augment_template = 'Sentence A: {{sentence1}}\nSentence B: {{sentence2}}\n\n' + augment_description
        if augment_template.find('the word') != -1:
            augment_template = augment_template.replace('the word', '"{{word}}"')
        elif augment_template.find('The word') != -1:
            augment_template = augment_template.replace('The word', '"{{word}}"')
        else:
            augment_template = augment_template.replace('word', '"{{word}}"')
        augment_template = augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['same_sense']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = 'Sentence A: {{sentence1}}\nSentence B: {{sentence2}}\n\n' + augment_template
        augment_template = augment_template + '\n||| {% if label != -1 %}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['GPT-3-prompt', 'GPT-3-prompt-with-label']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = '{{sentence1}}\n{{sentence2}}\n' + augment_template
        augment_template = augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['question-context']:
        augment_template = augment_description.replace('word', 'word \'{{word}}\'')
        augment_template = augment_template + '\n{{sentence1}}\n{{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['polysemous']:
        augment_template = augment_description.replace('word', 'word "{{word}}"')
        augment_template = augment_template + '\n\nSentence 1: {{sentence1}}\nSentence 2: {{sentence2}}\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'
    elif template_name in ['similar-sense']:
        if augment_description.find('the word') != -1:
            augment_template = augment_description.replace('the word', '{{word}}')
        else:
            augment_template = augment_description.replace('word', '{{word}}')
        augment_template = '{{sentence1}}\n{{sentence2}}\n' + augment_template + '\n||| {% if label != -1%}\n{{answer_choices[label]}}\n{% endif %}'

    # winogrande
    elif template_name in ['does underscore refer to']:
        if augment_description.find('the pronoun') != -1:
            augment_template = augment_description.replace('the pronoun', '_')
        else:
            augment_template = augment_description.replace('pronoun', '_')
        augment_template = augment_template + ' {{ option1 }} or {{ option2 }}? ||| {% if answer ==  "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
        augment_template = '{{ sentence }} ' + augment_template
    elif template_name in ['stand for']:
        augment_template = augment_description.replace('pronoun', '_')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template[:-len('the reference?')]
        else:
            augment_template = augment_template[:-len('reference?')]
        augment_template = augment_template + ' {{answer_choices[0]}} or {{answer_choices[1]}}?\n{{sentence}}|||\n{{answer_choices[answer | int - 1]}}'
    elif template_name in ['underscore refer to']:
        augment_template = augment_description.replace('pronoun', '_')
        augment_template = '{{sentence}}\n' + augment_template + ' {{ option1 }} or {{ option2 }}? ||| {% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    elif template_name in ['fill in the blank']:
        if augment_description.find('blanks') != -1:
            augment_template = augment_description.replace('blanks', '_')
        else:
            augment_template = augment_description.replace('blank', '_')
        augment_template = augment_template + '\n{{sentence}}\n\nChoices:\n- {{ option1 }}\n- {{ option2 }}\n\nAnswer: ||| {% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    elif template_name in ['True or False']:
        augment_template = augment_description.replace('pronoun', '_')
        if augment_template.find('the reference') != -1:
            augment_template = augment_template.replace('the reference', '{{option1}}')
        else:
            augment_template = augment_template.replace('reference', '{{option1}}')
        augment_template = augment_template + '\n{{sentence}}|||\n{{answer_choices[answer|int - 1]}}'
    elif template_name in ['Replace']:
        augment_template = augment_description.replace('pronoun', '_')
        augment_template = '{{sentence}}\n' + augment_template + '\n- {{option1}}\n- {{option2}}\n|||\n{% if answer == "1" %} {{option1}} {% else %} {{ option2 }} {% endif %}'
    else:
        print(f'无法识别的template_name: {template_name}')

    if augment_template is not None:
        augment_template = augment_template.strip()
        augment_template = augment_template.replace('  ', ' ')
        augment_template = augment_template.replace('  ', ' ')

    return augment_template


def read_input_template(args, input_dir, task_list):
    description_list = []
    config_dict = {}
    for task_name in task_list:
        name_tuple = task_name.split('/')
        if len(name_tuple) == 2:
            dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
        else:
            dataset_name, dataset_config_name = name_tuple[0], None

        config_file_path = os.path.join(input_dir, dataset_name)
        if dataset_config_name:
            config_file_path = os.path.join(config_file_path, dataset_config_name)
        config_file_path = os.path.join(config_file_path, 'templates.yaml')
        # 读该任务的templates
        task_templates = yaml.load(open(config_file_path, "r"), Loader=yaml.FullLoader)
        # 过滤非原始任务template
        templates_dict = task_templates['templates']
        config_dict[task_name] = task_templates
        for template in task_templates['templates'].values():
            description = get_description_from_template(task_name, template)
            if description is None:
                continue
            description_list.append(description)

    return config_dict, description_list


def build_output_config(args, origin_config_dict, all_description_list, output_entry_list):
    """根据dino生成的结果生成config, 这里主要是些"""

    augment_dict = {}
    for entry in output_entry_list:
        print(entry)
        text_a = entry.text_a
        text_b = entry.text_b

        if text_a not in augment_dict:
            augment_dict[text_a] = [text_b]
        else:
            augment_dict[text_a].append(text_b)

    # 生成新的config文件
    for task_name, config in origin_config_dict.items():

        templates_list = list(config['templates'].values())
        # 只用原始任务形式的template
        templates_list = [template_obj for template_obj in templates_list if template_obj.metadata.original_task is True]

        # 只用前N个
        # if args.input_dir != '/mfs/shaonan/moonshot/t-zero/templates':
        #     templates_list = templates_list[:args.Top_N_patterns]  # 当前任务的所有template obj

        # new_pattern_list = copy.deepcopy(patterns_list)
        new_template_dict = {}

        # 用于去重
        exist_description = set()
        exist_template = set()

        for template in templates_list:
            template_name = template.name
            option_list = template.answer_choices
            option_list = option_list.split('|||')

            origin_description = get_description_from_template(task_name, template)
            if origin_description is None:
                continue

            # OPTION: 是否不允许生成上一步一样的
            exist_description.add(origin_description)
            exist_template.add(template.jinja)

            # 该description所有augment的结果
            if origin_description in augment_dict:
                augment_result = augment_dict[origin_description]
            else:
                print(f'=========================')
                print(f'找不到augment的结果: {origin_description}')
                print(augment_dict)
                print(f'========================')
                continue

            # 规则过滤
            print('========filter before==========')
            print(f'origin_description: {origin_description}')
            print(augment_result)
            augment_result = [augment_description for augment_description in
                              augment_result if will_keep_with_clean_rule(task_name, template_name, augment_description, option_list)]
            print(f'========filter after==========')
            print(augment_result)

            for augment_description in augment_result:
                if augment_description in exist_description:
                    continue
                augment_template_text = build_template(task_name, template_name, augment_description)

                # description没有重复，但是转成template重复了，也不行
                if augment_template_text in exist_template:
                    continue

                new_template = copy.deepcopy(template)
                new_template.jinja = augment_template_text
                new_template.id = str(uuid.uuid4())
                new_template_dict[new_template.get_id()] = new_template

                exist_description.add(augment_description)
                exist_template.add(augment_template_text)

        # 生成新的template file，并输出
        new_config = copy.deepcopy(config)
        new_config['templates'] = dict()
        for template_id, template in new_template_dict.items():
            new_config['templates'][template_id] = template

        name_tuple = task_name.split('/')
        if len(name_tuple) == 2:
            dataset_name, dataset_config_name = name_tuple[0], name_tuple[1]
        else:
            dataset_name, dataset_config_name = name_tuple[0], None

        config_file_path = os.path.join(args.output_dir, dataset_name)
        if dataset_config_name:
            config_file_path = os.path.join(config_file_path, dataset_config_name)
        os.makedirs(config_file_path, exist_ok=True)
        config_file_path = os.path.join(config_file_path, 'templates.yaml')
        yaml.dump(new_config, open(config_file_path, "w"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_dir", type=str, required=True,
                        help="需要augment的template的目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--task_file", type=str,
                        default='/share/zongyu/shaonan/dino-main/task_specs/generate_task_description_en_v2.json',
                        help="A json file providing the instructions and other information required for dataset generation. "
                             "See the 'task_specs' directory for examples and 'README.md' for more details on how to create this file.")
    parser.add_argument("--task_list_file", type=str, default='/share/zongyu/shaonan/t-zero/config/setting_5/test.list')

    parser.add_argument("--model_name", type=str, default="/share/zongyu/shaonan/pretrained_model/gpt2-xl",
                        help="The pretrained model to use for dataset generation. Currently, only variants of GPT2 are supported.")

    parser.add_argument("--max_output_length", type=int, default=50,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--decay_constant", type=float, default=100,
                        help="The decay constant for self-debiasing")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--top_k", type=int, default=0,
                        help="k value for top-k sampling (set to 0 to perform no top-k sampling)")

    parser.add_argument("--num_entries_per_input_and_label", type=int, default=30,
                        help="The number of entries to generate for each pair of input text and label (only if --input_file is set)")
    parser.add_argument("--num_entries_per_label", type=int, default=None,
                        help="The number of entries to generate for each label (only if --input_file is not set)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--remove_duplicates", action='store_true',
                        help="Whether duplicates should be removed from the generated dataset")
    parser.add_argument("--remove_identical_pairs", action='store_true',
                        help="Whether text pairs with text_a == text_b should be removed from the dataset (only for text pair datasets)")

    parser.add_argument("--keep_outputs_without_eos", action='store_true',
                        help="If set to true, examples where the language model does not output a quotation mark ("
                             "which is interpreted as "
                             "a signal that it has completed its output) are not removed from the dataset.")
    parser.add_argument("--allow_newlines_in_outputs", action='store_true',
                        help="If set to true, model outputs that contain a newline character before the end-of-sequence token (a quotation "
                             "mark) are not removed from the dataset.")
    parser.add_argument("--min_num_words", type=int, default=-1,
                        help="The minimum number of (whitespace-separated) words for each dataset entry. Entries with fewer words are "
                             "removed.")
    parser.add_argument("--min_num_tokens", type=int, default=-1,
                        help="The minimum number of tokens for each dataset entry. Entries with fewer tokens are removed.")

    parser.add_argument("--Top_N_patterns", type=int, default=6,
                        help="使用top n 个pattern进行argument")

    # Miscellaneous further parameters
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    args.remove_identical_pairs = True
    # 没有用的参数，单纯防止报错
    args.input_file_type = 'plain'
    args.openai_api_key = None

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(args.task_file, 'r', encoding='utf8') as fh:
        # 提供instruction的文件
        task_specification = json.load(fh)
        # validate_task_spec(task_specification, with_inputs=args.input_file is not None)

    task_list = []
    with open(args.task_list_file, 'r') as f:
        for line in f.readlines():
            task_name = line.replace('\n', '')
            task_name = task_name.strip()
            task_list.append(task_name)

    # 根据input目录读需要augment的config文件
    print(f'Reading input config file: {args.input_dir}')
    config_dict, description_list = read_input_template(args, args.input_dir, task_list)

    inputs = description_list
    print(f'inputs: {inputs}')
    model = GPT2Wrapper(model_name=args.model_name,
                        use_cuda=not args.no_cuda) if not args.openai_api_key else args.model_name
    generator = DinoGenerator(
        task_spec=task_specification, model=model, openai_api_key=args.openai_api_key,
        max_output_length=args.max_output_length,
        decay_constant=args.decay_constant, top_p=args.top_p, top_k=args.top_k,
        remove_duplicates=args.remove_duplicates,
        remove_identical_pairs=args.remove_identical_pairs, min_num_words=args.min_num_words,
        min_num_tokens=args.min_num_tokens,
        keep_outputs_without_eos=args.keep_outputs_without_eos, allow_newlines_in_outputs=args.allow_newlines_in_outputs
    )

    print("Starting dataset generation with DINO...")
    outputs = generator.generate_dataset(inputs, num_entries_per_input_and_label=args.num_entries_per_input_and_label,
                                         num_entries_per_label=args.num_entries_per_label, batch_size=args.batch_size)
    # debug
    # with open(os.path.join(args.output_dir, 'augmented_prompt.txt'), 'w') as f:
    #     for entry in outputs:
    #         dict_line = {'text_a': entry.text_a, 'text_b': entry.text_b}
    #         f.write(json.dumps(dict_line) + '\n')

    build_output_config(args, config_dict, description_list, outputs)
