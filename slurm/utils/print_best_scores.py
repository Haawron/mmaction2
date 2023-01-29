from pathlib import Path
import re
import argparse
import pandas as pd
import json

from commons.patterns import (
    get_base_infodict_and_pattern_from_workdir_name,
    get_p_target_workdir_name_with_jid,
)


# test cases
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m cop
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m 'cop-from-k400'
    # 특이사항: validation, test가 없어서 log.json의 mca값으로 best를 판단해야 함
# python slurm/utils/print_best_scores.py -m vanilla
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m vanilla
# python slurm/utils/print_best_scores.py -d ucf2hmdb -b svt -m vanilla
# python slurm/utils/print_best_scores.py -m gcd4da
# python slurm/utils/print_best_scores.py -d ucf2hmdb -m gcd4da
# python slurm/utils/print_best_scores.py -d hmdb2ucf -b svt -m gcd4da
# python slurm/utils/print_best_scores.py -m gcd4da -db vanilla -p phase0 -t P02_P04 --abl one_way
# python slurm/utils/print_best_scores.py -m gcd4da -db vanilla -p phase0 -t P02_P04 --abl one_way -o



METRICS2DISPLAY = {
    'top1_acc': 'top1',
    'top5_acc': 'top5',
    'mean_class_accuracy': 'mca',
    'H_mean_class_accuracy': 'h', 'os*': 'os*',
    'sskmeans': 'h (sskmeans)',
    'recall_unknown': 'unk',
}
METRICS = METRICS2DISPLAY.values()

DATASETSHORTCUTS = {
    'u2h': 'ucf2hmdb',
    'h2u': 'hmdb2ucf',
    'b2k': 'babel2kinetics',
    'k2b': 'kinetics2babel',
}


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
    parser.add_argument('-smb', '--select-model-by', choices=METRICS, default='h (sskmeans)',
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
            'dataset': DATASETSHORTCUTS.get(args.dataset, args.dataset),
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
    df_list = [get_df_from_jid(jid, ignore_old_models=ignore_old_models) for jid in jids]
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
    p_target_workdir_cadidates = [p for p in p_target_workdir_parent.glob('**/*') if p.is_dir() and re.search(pattern, str(p))]
    for p_target_workdir in p_target_workdir_cadidates:
        info = get_best_info_by_target_workdir(p_target_workdir, select_model_by, ignore_old_models)
        if info:
            df = info2df(info)
            df_list.append(df)
    df = pd.concat(df_list)

    columns = [column for column in METRICS if column in df.columns]
    select_model_by = select_model_by if select_model_by in columns else 'mca'
    columns.remove(select_model_by)
    columns.insert(0, select_model_by)
    columns.append('jid')

    if model == 'vanilla':
        indices = ['dataset', 'backbone', 'model', 'task']
        sort_by = ['task', select_model_by]
        if dataset == 'ek100':
            indices.insert(3, 'domain')
            sort_by.insert(0, 'domain')
    elif model in ['gcd4da', 'cdar']:
        indices = ['dataset', 'backbone', 'model', 'phase', 'debias', 'ablation']
        # sort_by = ['debias', 'phase', select_model_by]
        sort_by = ['phase', select_model_by]
        if dataset == 'ek100':
            indices.insert(5, 'task')
            sort_by.insert(0, 'task')
    else:
        indices = ['dataset', 'backbone', 'model']
        sort_by = [select_model_by]
        if dataset == 'ek100':
            indices.append('task')
            sort_by.insert(0, 'task')

    ascending_columns = ['domain', 'task']
    ascending = [column in ascending_columns for column in sort_by]

    sort_ek100_task_by_target = lambda column: column.map(lambda value: ''.join(value.split('_')[::-1])) if column.name == 'task' else column
    sort_key_func = sort_ek100_task_by_target if dataset == 'ek100' else None

    with pd.option_context('display.max_colwidth', None, 'display.precision', 4):  # more options can be specified also
        print(
            df
            .reset_index()
            .set_index(indices)
            .sort_values(by=sort_by, ascending=ascending, key=sort_key_func)[columns]
            .rename(columns=str.upper)
            .rename_axis(index=str.capitalize)
            .fillna('-')
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
        best_info = max(infos, key=lambda info: info.get('select_model_by', info['mca']))
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
    p_target_workdir = get_p_target_workdir_name_with_jid(jid)
    return get_best_info_by_target_workdir(p_target_workdir, select_model_by)


def get_best_info_by_target_workdir(p_target_workdir, select_model_by='mca', ignore_old_models=False):
    def p_log2jid(p_log):
        jid = '_'.join(re.findall(r'/(\d+)__[\w-]+/(\d+)/', str(p_log))[0])
        return jid

    if p_target_workdir: # for a single job
        info, pattern = get_base_infodict_and_pattern_from_workdir_name(p_target_workdir)

        p_logs = list(p_target_workdir.glob('**/*.log'))
        # p_logs_and_score_dicts = [(p_log, get_test_scores_from_logfile(p_log, for_cop=('cop' in info['model']))) for p_log in p_logs]
        p_logs_and_score_dicts = [(p_log, get_test_scores_from_logfile(p_log, for_cop=False)) for p_log in p_logs]
        p_logs_and_score_dicts = [(p_log, score_dict) for p_log, score_dict in p_logs_and_score_dicts if score_dict]  # drop jobs with no score
        if ignore_old_models:
            p_logs_and_score_dicts = [(p_log, score_dict) for p_log, score_dict in p_logs_and_score_dicts if p_log2jid(p_log) >= 19000]
        if p_logs_and_score_dicts:
            p_logs, score_dicts = zip(*p_logs_and_score_dicts)
            arg_best = max(range(len(score_dicts)), key=lambda i: score_dicts[i].get(select_model_by, score_dicts[i].get('h', score_dicts[i].get('mca', 0))))
            p_log = p_logs[arg_best]
            jid = p_log2jid(p_log)

            # print(pattern, '\n', str(p_log), sep='')  # for debugging
            info = re.search(pattern, str(p_log)).groupdict()
            info.update(score_dicts[arg_best])
            info['jid'] = jid
            # info['ckpt'] = str([p for p in p_log.parent.glob('*.pth') if ('latest' if ('cop' in info['model']) else 'best') in str(p)][0])
            info['ckpt'] = str([p for p in p_log.parent.glob('*.pth') if 'best' in str(p)][0])
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

    # for ucf2hmdb, hmdb2ucf, babel2kinetics, kinetics2babel
        # linear_probe: p_train_workdirs / dataset / backbone / model / probed_on  # not implemented
        # vanilla:      p_train_workdirs / dataset / backbone / model / task
        # gcd4da, cdar: p_train_workdirs / dataset / backbone / model / debias / phase / ablation
        # else:         p_train_workdirs / dataset / backbone / model

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
        matched = re.split(r'Testing results of the best checkpoint\s', data)
        if len(matched) == 1:
            return None
        matched_scores = re.findall(r'([\w\*]+): (\d.\d+)', matched[-1])  # 0 <= score <= 1
        test_scores = {METRICS2DISPLAY[name]: float(score) for name, score in matched_scores}
        matched_scores = re.findall(r'([\w\*]+): (\d\d.\d+)', matched[-1])  # 0 <= score(percent) <= 100
        test_scores.update({METRICS2DISPLAY[name]: float(score)/100 for name, score in matched_scores})
    return test_scores


def info2df(info: dict):
    del info['ckpt']
    del info['config']
    df = pd.DataFrame.from_dict([info])
    return df


if __name__ == '__main__':
    main()
