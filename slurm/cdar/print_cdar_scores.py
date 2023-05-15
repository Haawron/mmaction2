from pathlib import Path
import re
import os
import argparse
import socket

import numpy as np
import pandas as pd


only_best_children = True
latex_view = False

dispmap = {
    'gcd_v2': 'ALL',
    'gcd_v2_old': 'Old',
    'gcd_v2_new': 'New',
    'gcd_v2_balanced': 'ALL(B)',
    'gcd_v2_balanced_old': 'Old(B)',
    'gcd_v2_balanced_new': 'New(B)',
    'kmeans_balanced': 'ALL(kmeans,B)',
    'kmeans_balanced_old': 'Old(kmeans,B)',
    'kmeans_balanced_new': 'New(kmeans,B)',
    'kmeans': 'ALL(kmeans)',
    'kmeans_old': 'Old(kmeans)',
    'kmeans_new': 'New(kmeans)',
}


parser = argparse.ArgumentParser('')
parser.add_argument('-p', '--show-path', action='store_true')
parser.add_argument('-P', '--parsable', action='store_true')
parser.add_argument('-k', '--show-kmeans', action='store_true')
parser.add_argument('-v', '--in-pivot', action='store_true')
args = parser.parse_args()

print_args = ['display.precision', 1, 'max_colwidth', None]
if args.parsable:
    print_args += ['display.multi_sparse', False]

orders = {
    'task': {},
    'subtask': {},
    'model': {},
}

# TODO: 다른 사람이 실행했을 때
p = Path(f'/data/hyogun/repos/haawron_mmaction2/work_dirs/train_output/cdar')
records = []
for p_pkl in p.rglob('best_pred.pkl'):
    task, subtask, model, add_on, extra_setting, jobname, job_array_idx, *_ = p_pkl.parts[p_pkl.parts.index('cdar')+1:]
    order, task = task.split('_', 1)
    orders['task'][task] = str(order)  # will be sorted by dictionary order rather than arithmetic order
    order, subtask = subtask.split('_', 1)
    if 'closed' in subtask:
        continue
    orders['subtask'][subtask] = str(order)
    order, model = model.split('_', 1)
    orders['model'][model] = str(order)
    settings = {
        'task': task,
        'subtask': subtask,
        'model': model,
        'option1': add_on,
        'option2': extra_setting
    }
    p_log = next(p_pkl.parent.glob('*.log'))
    with p_log.open() as f:
        log = f.read()
        matched = list(re.finditer('Testing results of the best checkpoint', log))
        if len(matched) == 0:
            continue
        else:
            matched = matched[0]
        log_trunc = log[matched.end(0):]
        found = {metric: 100*float(score) for metric, score in re.findall(r'(\w+): (\d+\.\d+)', log_trunc)}
    record = {
        **settings, **found,
        'jid': jobname.split('__')[0] + f'_{job_array_idx}'
    }
    if args.show_path:
        record['path'] = p_log.parent
    records.append(record)
df = pd.DataFrame.from_records(records)

if latex_view or only_best_children:
    df['parent_jid'] = df['jid'].map(lambda jid: jid.split('_')[0])
    df = df.sort_values(['gcd_v2_balanced'], ascending=False).groupby('parent_jid').head(1)
    del df['parent_jid']

# todo: LateX setting이면 group by model 해서 best 뽑고
# subtask를 column으로 보내고 multilevel로 만들기
# task 마다 mean 구해서 column 추가

sort_bys = ['task', 'subtask', 'model']
for sort_by in sort_bys:
    df[sort_by+'_order'] = df[sort_by].map(orders[sort_by])

sort_by_metrics = [
    'gcd_v2_balanced', 'gcd_v2_balanced_old', 'gcd_v2_balanced_new',
    'gcd_v2', 'gcd_v2_old', 'gcd_v2_new',
]

if args.in_pivot:
    df = df.sort_values(['gcd_v2_balanced'], ascending=False).groupby(['task', 'subtask', 'model']).head(1)
    for task in df['task'].unique():
        print(task)
        df_task = df[df['task']==task]
        df_task = df_task.set_index('model')
        df_task = df_task.sort_values(
            by=list(map(lambda s: f'{s}_order', sort_bys)) + sort_by_metrics + ['jid'],
            ascending=[True]*len(sort_bys)+[False]*len(sort_by_metrics)+[True]).fillna('-')
        # reshaping data
        df_task = df_task.reset_index()
        df_task = pd.melt(
            df_task,
            id_vars=['model', 'task', 'subtask', 'option1', 'option2'],
            value_vars=sort_by_metrics[:3],
            var_name='metric'
        )  # unpivot
        df_average = df_task.pivot_table(values='value', index=['model'], columns=['metric'], aggfunc=np.mean, sort=False)
        df_average = pd.concat([df_average], keys=['Average' + ' '*(len(df_task['subtask'][0])-len('ALL(B)'))], names=['subtask'], axis=1)  # insert level 0 column
        df_task = df_task.pivot_table(values='value', index=['model'], columns=['subtask', 'metric'], sort=False)
        df_task = df_task.sort_index(key=lambda models: models.map(orders['model']))
        df_task = pd.concat([df_task, df_average], axis=1).fillna('-')
        df_task = df_task.rename(columns=lambda s: s.replace('_', ' → '), level=0)
        df_task = df_task.rename(columns=dispmap, level=1)
        with pd.option_context(*print_args):
            print(df_task)
        print()
        print()

else:
    df = df.set_index(['task', 'subtask', 'model', 'option1', 'option2'])
    df = df.sort_values(
        by=list(map(lambda s: f'{s}_order', sort_bys)) + sort_by_metrics + ['jid'],
        ascending=[True]*len(sort_bys)+[False]*len(sort_by_metrics)+[True]).fillna('-')

    for sort_by in sort_bys:
        del df[f'{sort_by}_order']

    cols = [
        'gcd_v2_balanced', 'gcd_v2_balanced_old', 'gcd_v2_balanced_new',
        'gcd_v2', 'gcd_v2_old', 'gcd_v2_new']
    if args.show_kmeans:
        cols += [
            'kmeans_balanced', 'kmeans_balanced_old', 'kmeans_balanced_new',
            'kmeans', 'kmeans_old', 'kmeans_new',
        ]
    if args.show_path:
        cols += ['path']
    else:
        cols += ['jid']
    df = df[cols]
    df = df.rename(columns=dispmap)
    with pd.option_context(*print_args):
        print(df)
