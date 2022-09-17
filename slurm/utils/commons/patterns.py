from typing import Union, Tuple

from pathlib import Path
import re
import argparse


def get_full_infodict_from_jid(jid:Union[str,int]):
    p_target_workdir = get_p_target_workdir_name_with_jid(jid)
    info = get_infodict_from_workdir_name(p_target_workdir)
    source, target = __get_source_and_target_from_infodict(info)
    info.update({
        'source': source,
        'target': target,
    })

    _, array_idx = _parse_jid(jid)
    if array_idx:  # if array job
        info['ckpt'] = str([p for p in p_target_workdir.glob('**/*.pth') if 'best' in str(p)][0] or None)  # todo: how to deal with None
        info['config'] = str([p for p in p_target_workdir.glob('**/*.py')][0])
    return info


def get_infodict_from_workdir_name(workdir_name:Union[str,Path]):  # workdir이 얼마나 specific한지 몰라서 full info는 못 얻음
    workdir_name = str(workdir_name)
    base_info, pattern = get_base_infodict_and_pattern_from_workdir_name(workdir_name)
    info = re.search(pattern, workdir_name).groupdict()
    info.update(base_info)
    return info


def get_base_infodict_and_pattern_from_workdir_name(workdir_name:Union[str,Path]):
    workdir_name = str(workdir_name)
    pattern = r'work_dirs/train_output/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/'
    try:
        base_info = re.search(pattern, workdir_name).groupdict()
    except AttributeError as e:
        print(e)
        print(f'\nworkdir name: {workdir_name}')
        print(f'pattern: {pattern}')
        exit(1)

    if base_info['dataset'] == 'ek100' and base_info['model'] == 'vanilla':
        pattern += r'(?P<domain>[\w-]+)/'
    elif 'gcd4da' in base_info['model'] or 'cdar' in base_info['model']:
        pattern += r'(?P<debias>[\w-]+)/(?P<phase>[\w-]+)/(?P<ablation>[\w-]+)/'
    elif 'osbp' in base_info['model']:
        pattern += r'(?P<debias>[\w-]+)/'

    if base_info['dataset'] == 'ek100' or base_info['model'] == 'vanilla':
        pattern += r'(?P<task>[\w-]+)/'

    pattern += r'\d+__'

    return base_info, pattern


def get_p_target_workdir_name_with_jid(jid:Union[str,int]) -> Path:
    jid, array_idx = _parse_jid(jid)
    p_train_workdir = Path('work_dirs/train_output')
    pattern = rf'/{jid}__[\w-]+'
    if array_idx is not None:
        pattern += rf'/{array_idx}/\d{{8}}-\d{{6}}$'
    pattern += r'$'
    p_target_workdir, = [
        p for p in p_train_workdir.glob('**/*')
        if re.findall(pattern, str(p))
    ] or [None]
    return p_target_workdir


def __get_source_and_target_from_infodict(info:dict):
    if info['dataset'] == 'ek100':
        if 'vanilla' in info['model']:
            source = info['domain']
            target = None  # todo
        else:
            source, target = info['task'].split('_')
    else:
        source, target = info['dataset'].split('2')

    return source, target


def _parse_jid(jid:Union[str,int]) -> Tuple[str,Union[str,None]]:
    if type(jid) == str and '_' in jid:  # if array job
        jid, array_idx = jid.split('_')
    else:
        jid = str(jid)
        array_idx = None
    return jid, array_idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('queries', type=str, nargs='*', default=['source', 'target', 'ckpt'])
    parser.add_argument('-d', '--workdir-name', type=str, default=None)
    parser.add_argument('-j', '--jid', type=str, default='5504_1')
    args = parser.parse_args()

    assert not (args.jid and args.workdir_name), f''

    if args.jid:
        info = get_full_infodict_from_jid(args.jid)
    else:
        info = get_infodict_from_workdir_name(args.wordir_name)

    values = []
    for query in args.queries:
        assert query in info, f'Wrong query: {query}\nAvailable queries: {list(info.keys())}'
        value = info[query]
        values.append(value)

    print(' '.join(values))
