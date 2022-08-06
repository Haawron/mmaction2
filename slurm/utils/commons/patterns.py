from typing import Union

from pathlib import Path
import re


def get_infodict_and_pattern_from_workdir_name(workdir_name:Union[str,Path]):
    workdir_name = str(workdir_name)
    pattern = r'work_dirs/train_output/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/'
    base_info = re.search(pattern, workdir_name).groupdict()

    if base_info['dataset'] == 'ek100' and 'vanilla' in base_info['model']:
        pattern += r'(?P<domain>[\w-]+)/'
    elif 'gcd4da' in base_info['model'] or 'cdar' in base_info['model']:
        pattern += r'(?P<debias>[\w-]+)/(?P<phase>[\w-]+)/(?P<ablation>[\w-]+)/'
    
    if base_info['dataset'] == 'ek100' or 'vanilla' in base_info['model']:
        pattern += r'(?P<task>[\w-]+)/'

    pattern += r'\d+__'
    
    return base_info, pattern
