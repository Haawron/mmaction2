from pathlib import Path
import re
import argparse
import pandas as pd
import json

# test cases
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m cop
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m 'cop-from-k400'
    # 특이사항: validation, test가 없어서 log.json의 mca값으로 best를 판단해야 함
# python slurm/utils/print_best_scores.py -m gcd4da -p phase0 -t P02_P04 --abl one_way
# python slurm/utils/print_best_scores.py -m gcd4da -p phase0 -t P02_P04 --abl one_way -o


def main():
    """
    if given a jid (then the target workdir would be for a single job array)
        - get all logs (multiple sub-jobs may be in the dir)
        - find the best log's path which has the best test acc in the log
        - print [JID Dataset Backbone Model Domain Task Acc]

    elif given config vars
        - make workdir_path with given config vars
        - find all possible products of not-given config vars
        - for each product
            - make full path to the workdir
            - and get paths to job array dirs
                - for each job array dir
                    - do the same thing in the if-block above
    """
    parser = argparse.ArgumentParser(description='Print the Best Model\'s Info, Given JID Or Config Vars.')
    parser.add_argument('-o', '--one-line', action='store_true',
                        help='')
    parser.add_argument('-ig', '--ignore-old-models', action='store_true',
                        help='ignore models with $jid < 19000')
    parser.add_argument('-smb', '--select-model-by', choices=['acc', 'mca'], default='mca',
                        help='')

    parser.add_argument('-j', '--jids', default=None, nargs='+',
                        help='The main job ids of the task')

    parser.add_argument('-d', '--dataset', default='ek100',
                        help='')
    parser.add_argument('-b', '--backbone', default='tsm',
                        help='')
    parser.add_argument('-m', '--model', default=None,
                        help='')
    parser.add_argument('-dom', '--domain', default=None,
                        help='')
    parser.add_argument('-db', '--debias', default=None,
                        help='')
    parser.add_argument('-ph', '--phase', default=None,
                        help='')
    parser.add_argument('-t', '--task', default=None,
                        help='')
    parser.add_argument('-abl', '--ablation', default=None,
                        help='')

    parser.add_argument('--test', action='store_true',
                        help='')

    args = parser.parse_args()

    if args.test:
        # for exp_dict in get_test_cases():
        #     print_info_from_config_vars_for_test(exp_dict)
        raise NotImplementedError
    
    if args.jids:
        if args.one_line:
            assert len(args.jids) == 1
            print_info_from_jid_for_test(args.jids[0])
        else:
            print_df_from_jids(args.jids, ignore_old_models=args.ignore_old_models)
    elif args.model:
        exp_dict = {
            'dataset': args.dataset,
            'backbone': args.backbone,
            'model': args.model,
            'domain': args.domain,
            'debias': args.debias,
            'phase': args.phase,
            'ablation': args.ablation,
            'task': args.task,
        }
        if args.one_line:
            print_info_from_config_vars_for_test(**exp_dict, select_model_by=args.select_model_by, ignore_old_models=args.ignore_old_models)
        else:
            print_df_from_config_vars(**exp_dict, select_model_by=args.select_model_by, ignore_old_models=args.ignore_old_models)


def print_df_from_jids(jids, ignore_old_models=False):
    def get_df_from_jid(jid, ignore_old_models=False):
        if ignore_old_models and int(jid) < 19000:
            return None
        info = get_best_info_from_jid(jid)
        if info:
            return info2df(info)
        else:
            return None
    df_list = [get_df_from_jid(jid, ignore_old_models=False) for jid in jids]
    df_list = [df for df in df_list if df is not None]  # drop None(cannot found the best model's pth)
    df = pd.concat(df_list)
    with pd.option_context('display.max_colwidth', None):  # more options can be specified also
        print(df)


def print_df_from_config_vars(
    dataset,
    backbone,
    model,
    domain=None,
    debias=None,
    phase=None,
    ablation=None,
    task=None,
    select_model_by='mca', ignore_old_models=False
):
    p_target_workdir_parent = get_validated_p_target_workdir(dataset, backbone, model, domain, debias, phase, ablation, task)
    pattern = r'\d+__[^/]*$'
    df_list = []
    for p_target_workdir in [p for p in p_target_workdir_parent.glob('**/*') if p.is_dir() and re.search(pattern, str(p))]:
        info = get_best_info_by_target_workdir(p_target_workdir, select_model_by, ignore_old_models)
        if info:
            df = info2df(info)
            df_list.append(df)
    df = pd.concat(df_list)
    sort_ek100_task_by_target = lambda column: column.map(lambda value: ''.join(value.split('_')[::-1])) if column.name == 'Task' else column
    with pd.option_context('display.max_colwidth', None):  # more options can be specified also
        if model == 'vanilla':
            indices = ['Dataset', 'Backbone', 'Model', 'Domain', 'Task']
            sort_by = ['Domain', 'Task', select_model_by.upper()]
            ascending = [True, True, False]
            columns = ['ACC', 'MCA', 'JID'] if select_model_by == 'acc' else ['MCA', 'ACC', 'JID']
        elif model in ['gcd4da', 'cdar']:
            indices = ['Dataset', 'Backbone', 'Model', 'Debias', 'Phase', 'Task', 'Ablation']
            sort_by = ['Task', 'Ablation', select_model_by.upper()]
            ascending = [True, True, False]
            columns = ['ACC', 'MCA', 'UNK', 'JID'] if select_model_by == 'acc' else ['MCA', 'ACC', 'UNK', 'JID']
            if phase and phase == 'phase0':
                columns.remove('UNK')
        else:
            indices = ['Dataset', 'Backbone', 'Model', 'Task']
            sort_by = ['Task', select_model_by.upper()]
            ascending = [True, False]
            columns = ['ACC', 'MCA', 'UNK', 'JID'] if select_model_by == 'acc' else ['MCA', 'ACC', 'UNK', 'JID']
            if any(kw in model for kw in ['cop', 'dann']): 
                columns.remove('UNK')
        
        sort_key_func = sort_ek100_task_by_target if dataset == 'ek100' else None
        
        if dataset != 'ek100':
            if model != 'vanilla':
                indices.remove('Task')
                if 'Task' in sort_by:
                    sort_by.remove('Task')
                    ascending.remove(True)
            else:
                indices.remove('Domain')
                sort_by.remove('Domain')
                ascending.remove(True)

        print(
            df
            .reset_index()
            .set_index(indices)
            .sort_values(by=sort_by, ascending=ascending, key=sort_key_func)[columns]
        )

def print_info_from_jid_for_test(jid):
    info = get_best_info_from_jid(jid)
    if info:
        print(' '.join(map(str, info.values())))
    else:
        exit(1)


def print_info_from_config_vars_for_test(
        dataset,
        backbone,
        model,

        domain,
        debias, phase, ablation,
        
        task,
        select_model_by='mca', ignore_old_models=False
    ):
    p_target_workdir_parent = get_validated_p_target_workdir(dataset, backbone, model, domain, debias, phase, ablation, task)
        
    pattern = r'\d+__[^/]*$'
    infos = []
    for p_target_workdir in [p for p in p_target_workdir_parent.glob('**/*') if p.is_dir() and re.search(pattern, str(p))]:
        info = get_best_info_by_target_workdir(p_target_workdir, select_model_by, ignore_old_models)
        if info:
            infos.append(info)
    if infos:
        #  input                   output
        # [captured_from_path      'acc', 'mca', 'jid', 'ckpt', 'config']
        best_info = max(infos, key=lambda info: info[select_model_by])
        print(' '.join(map(str, best_info.values())))
    else:
        exit(1)
    

def get_best_info_from_jid(jid, select_model_by='mca'):
    """
    Returns:
        jid, acc, source, target
        Or
        jid, acc, domain, {"source-only" or "target-only"}
    """
    p_train_workdirs = Path(r'work_dirs/train_output')
    p_target_workdir = get_p_target_workdir_with_jid(p_train_workdirs, jid)
    return get_best_info_by_target_workdir(p_target_workdir, select_model_by)


def get_best_info_by_target_workdir(p_target_workdir, select_model_by='mca', ignore_old_models=False):
    def p_log2jid(p_log):
        jid = '_'.join(re.findall(r'/(\d+)__[\w-]+/(\d+)/', str(p_log))[0])
        return jid

    if p_target_workdir: # for a single job
        pattern = r'work_dirs/train_output/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/'
        info = re.search(pattern, str(p_target_workdir)).groupdict()

        p_logs = list(p_target_workdir.glob('**/*.log'))
        p_logs_and_score_dicts = [(p_log, get_test_scores_from_logfile(p_log, for_cop=('cop' in info['model']))) for p_log in p_logs]
        p_logs_and_score_dicts = [(p_log, score_dict) for p_log, score_dict in p_logs_and_score_dicts if score_dict]  # drop jobs with no score
        if ignore_old_models:
            p_logs_and_score_dicts = [(p_log, score_dict) for p_log, score_dict in p_logs_and_score_dicts if p_log2jid(p_log) >= 19000]
        if p_logs_and_score_dicts:
            p_logs, score_dicts = zip(*p_logs_and_score_dicts)
            arg_best = max(range(len(score_dicts)), key=lambda i: score_dicts[i][select_model_by])
            p_log = p_logs[arg_best]
            jid = p_log2jid(p_log)

            if info['dataset'] == 'ek100' and 'vanilla' in info['model']:
                pattern += r'(?P<domain>[\w-]+)/'
            elif 'gcd4da' in info['model'] or 'cdar' in info['model']:
                pattern += r'(?P<debias>[\w-]+)/(?P<phase>[\w-]+)/(?P<ablation>[\w-]+)/'
            
            if info['dataset'] == 'ek100' or 'vanilla' in info['model']:
                pattern += r'(?P<task>[\w-]+)/'

            pattern += r'\d+__'
            
            info = re.search(pattern, str(p_log)).groupdict()
            info.update(score_dicts[arg_best])
            info['jid'] = jid
            info['ckpt'] = str([p for p in p_log.parent.glob('*.pth') if ('latest' if ('cop' in info['model']) else 'best') in str(p)][0])
            info['config'] = str(next(p_log.parent.glob('*.py')))
            return info
    return None


def get_validated_p_target_workdir(
    dataset,
    backbone,
    model,

    domain,  # for vanilla
    debias, phase, ablation,  # for gcd4da

    task,
    one_line=False,
):
    p_train_workdirs = Path(r'work_dirs/train_output')
    assert dataset and backbone and model

    p_target_workdir_parent = p_train_workdirs / dataset / backbone / model

    # for ek100
        # vanilla: p_train_workdirs / dataset / backbone / model / domain / task
        # gcd4da:  p_train_workdirs / dataset / backbone / model / debias / phase / ablation / task
        # else:    p_train_workdirs / dataset / backbone / model / task

    # for ucf2hmdb, hmdb2ucf, ...
        # vanilla: p_train_workdirs / dataset / backbone / model / task
        # gcd4da:  p_train_workdirs / dataset / backbone / model / debias / phase / ablation
        # else:    p_train_workdirs / dataset / backbone / model

    if model == 'vanilla':
        if one_line:
            assert domain, 'vanilla needs domain for one-line mode'
        assert not (phase or ablation), '`phase` nor `ablation` should not be provided with `domain`'
        if domain:  # only for ek100
            p_target_workdir_parent /= domain
    elif model in ['gcd4da', 'cdar']:
        if one_line:
            assert debias and phase and ablation, 'provide both `debias`, `phase` and `ablation` for one-line mode'
        else:
            assert not phase or debias  # if phase then debias
            assert not ablation or phase  # if ablation then phase
        if debias:
            p_target_workdir_parent /= debias
        if phase:
            p_target_workdir_parent /= phase
        if ablation:
            p_target_workdir_parent /= ablation

    if task:
        p_target_workdir_parent /= task

    assert p_target_workdir_parent.is_dir(), f'{str(p_target_workdir_parent)} is not a directory'
    return p_target_workdir_parent


def get_p_target_workdir_with_jid(p_workdirs, jid: str):
    p_target_workdir, = [
        p for p in p_workdirs.glob('**/*')
        if (jid in p.stem) and ('test_output' not in str(p)) and p.is_dir()
    ] or [None]
    return p_target_workdir


def get_test_scores_from_logfile(p_log, for_cop=False):
    if for_cop:
        with p_log.with_name(p_log.name + '.json').open('r') as f:
            data = f.read()
            string = '{ "data": [' + ','.join(data.strip().split('\n')) + '] }'
            data = json.loads(string)['data'][-1]  # last train log
        test_scores = {
            'acc': data['top1_acc'],
            'mca': data['mca'],
            'unk': None,
        }
    else:
        with p_log.open('r') as f:
            data = f.read()
        # [\w\W]: work-around for matching all characters including new-line
        pattern = r'Testing results of the best checkpoint[\w\W]*top1_acc: (?P<acc>[\d\.]+)[\w\W]*mean_class_accuracy: (?P<mca>[\d\.]+)([\w\W]*recall_unknown: (?P<unk>[\d\.]+))?'
        found = re.search(pattern, data)
        if found:
            test_scores = found.groupdict()
        else:  # still training or failed
            test_scores = None
    return test_scores


def info2df(info: dict):
    del info['ckpt']
    del info['config']
    # boilerplates to convert info-dict to df
    all_capital = ['jid', 'acc', 'mca', 'unk']
    info = {k.capitalize() if k not in all_capital else k.upper(): [v] for k, v in info.items()}
    df = pd.DataFrame.from_dict(info)
    df = df.set_index('JID')
    return df
        

if __name__ == '__main__':
    main()
