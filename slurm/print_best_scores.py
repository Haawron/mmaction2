from pathlib import Path
import re
import argparse
import pandas as pd


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
    parser.add_argument('-t', '--task', default=None,
                        help='')

    args = parser.parse_args()
    
    if args.jids:
        if args.one_line:
            assert len(args.jids) == 1
            print_info_from_jid_for_test(args.jids[0])
        else:
            print_df_from_jids(args.jids)
    elif args.dataset and args.model:
        if args.one_line:
            print_info_from_config_vars_for_test(args.dataset, args.backbone, args.model, args.domain, args.task)
        else:
            print_df_from_config_vars(args.dataset, args.backbone, args.model, args.domain, args.task, args.select_model_by)


def print_df_from_jids(jids):
    def get_df_from_jid(jid):
        info = get_best_info_from_jid(jid)
        if info:
            return info2df(info)
        return None
    df_list = [get_df_from_jid(jid) for jid in jids]
    df_list = [df for df in df_list if df is not None]  # drop None(cannot found the best model's pth)
    df = pd.concat(df_list)
    with pd.option_context('display.max_colwidth', None):  # more options can be specified also
        print(df)


def print_df_from_config_vars(dataset, backbone, model, domain=None, task=None, select_model_by='mca'):
    p_train_workdirs = Path(r'work_dirs/train_output')
    p_target_workdir_parent = p_train_workdirs / dataset / backbone / model
    if domain and model == 'vanilla':
        p_target_workdir_parent /= domain
    if task:
        p_target_workdir_parent /= task
    pattern = r'\d+__[^/]*$'
    df_list = []
    for p_target_workdir in [p for p in p_target_workdir_parent.glob('**/*') if p.is_dir() and re.search(pattern, str(p))]:
        info = get_best_info_by_target_workdir(p_target_workdir, select_model_by)
        if info:
            df = info2df(info)
            df_list.append(df)
    df = pd.concat(df_list)
    with pd.option_context('display.max_colwidth', None):  # more options can be specified also
        if model == 'vanilla':
            indices = ['Dataset', 'Backbone', 'Model', 'Domain', 'Task']
            sort_by = ['Domain', 'Task', select_model_by.upper()]
            ascending = [True, True, False]
            columns = ['ACC', 'MCA', 'JID'] if select_model_by == 'acc' else ['MCA', 'ACC', 'JID']
            key = None
        else:
            indices = ['Dataset', 'Backbone', 'Model', 'Task']
            sort_by = ['Task', select_model_by.upper()]
            ascending = [True, False]
            columns = ['ACC', 'MCA', 'JID'] if select_model_by == 'acc' else ['MCA', 'ACC', 'JID']
            key = lambda column: column.map(lambda value: ''.join(value.split('_')[::-1])) if column.name == 'Task' else column  # if task, sort by its target domain
        print(
            df
            .reset_index()
            .set_index(indices)
            .sort_values(by=sort_by, ascending=ascending, key=key)[columns]
        )

def print_info_from_jid_for_test(jid):
    info = get_best_info_from_jid(jid)
    if info:
        # if vanilla: ['dataset', 'backbone', 'vanilla', 'domain', 'task', 'acc', 'mca', 'jid', 'ckpt', 'config']
        # else:       ['dataset', 'backbone', 'model',             'task', 'acc', 'mca', 'jid', 'ckpt', 'config']
        print(' '.join(map(str, info.values())))
    else:
        exit(1)


def print_info_from_config_vars_for_test(dataset, backbone, model, domain, task, select_model_by='mca'):
    assert dataset and backbone and model and domain and task
    p_train_workdirs = Path(r'work_dirs/train_output')
    p_target_workdir_parent = p_train_workdirs / dataset / backbone / model / domain / task
    pattern = r'\d+__[^/]*$'
    infos = []
    for p_target_workdir in [p for p in p_target_workdir_parent.glob('**/*') if p.is_dir() and re.search(pattern, str(p))]:
        info = get_best_info_by_target_workdir(p_target_workdir, select_model_by)
        infos.append(info)
    if infos:
        # if vanilla: ['dataset', 'backbone', 'vanilla', 'domain', 'task', 'acc', 'mca', 'jid', 'ckpt', 'config']
        # else:       ['dataset', 'backbone', 'model',             'task', 'acc', 'mca', 'jid', 'ckpt', 'config']
        best_info = max(infos, key=lambda info: info[select_model_by])
        print(' '.join(map(str, best_info.values())))
    else:
        exit(1)
    

def get_best_info_from_jid(jid, select_model_by='mca'):
    """The structure of the target workdir would be

        if domain adapted:
            p_workdirs / dataset / backbone / model / {source}__{target} / {jid}__{jobname} /
                0/{time_job_submitted}/
                    {time_training_begins}.log
                1/...

        else if source/target-only:
            p_workdirs / dataset / backbone / model / domain / {"source-only" or "target-only"} / {jid}__{jobname} /
                0/...
                1/...

        Returns:
            jid, acc, source, target
            Or
            jid, acc, domain, {"source-only" or "target-only"}
    """
    p_train_workdirs = Path(r'work_dirs/train_output')
    p_target_workdir = get_p_target_workdir_with_jid(p_train_workdirs, jid)
    return get_best_info_by_target_workdir(p_target_workdir, select_model_by)


def get_best_info_by_target_workdir(p_target_workdir, select_model_by='mca'):
    if p_target_workdir: # for a single job
        p_logs = list(p_target_workdir.glob('**/*.log'))
        p_logs_and_score_dicts = [(p_log, get_test_scores_from_logfile(p_log)) for p_log in p_logs]
        p_logs_and_score_dicts = [(p_log, score_dict) for p_log, score_dict in p_logs_and_score_dicts if score_dict]  # drop jobs with no score
        if p_logs_and_score_dicts:
            p_logs, score_dicts = zip(*p_logs_and_score_dicts)
            arg_best = max(range(len(score_dicts)), key=lambda i: score_dicts[i][select_model_by])
            p_log = p_logs[arg_best]
            jid = int(re.findall(r'/(\d+)__', str(p_log))[0])
            if 'only' in str(p_log):
                pattern = r'/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/(?P<domain>[\w-]+)/(?P<task>[\w-]+)/\d+__'
            else:
                pattern = r'/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/(?P<task>[\w-]+)/\d+__'
            m = re.search(pattern, str(p_log))
            info = m.groupdict()
            info.update(score_dicts[arg_best])
            info['jid'] = jid
            info['ckpt'] = str([p for p in p_log.parent.glob('*.pth') if 'best' in str(p)][0])
            info['config'] = str(next(p_log.parent.glob('*.py')))
            return info
    return None


def get_p_target_workdir_with_jid(p_workdirs, jid: str):
    p_target_workdir, = [
        p for p in p_workdirs.glob('**/*')
        if (jid in p.stem) and ('test_output' not in str(p)) and p.is_dir()
    ] or [None]
    return p_target_workdir


def get_test_scores_from_logfile(p_log):
    with p_log.open('r') as f:
        data = f.read()
    # [\w\W]: work-around for matching all characters including new-line
    pattern = r'Testing results of the best checkpoint[\w\W]*top1_acc: (?P<acc>[\d\.]+)[\w\W]*mean_class_accuracy: (?P<mca>[\d\.]+)'
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
    all_capital = ['jid', 'acc', 'mca']
    info = {k.capitalize() if k not in all_capital else k.upper(): [v] for k, v in info.items()}
    df = pd.DataFrame.from_dict(info)
    df = df.set_index('JID')
    return df
        

if __name__ == '__main__':
    main()
