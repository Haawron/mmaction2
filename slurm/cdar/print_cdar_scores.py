from pathlib import Path
import re

import pandas as pd


only_best_children = True
latex_view = False


orders = {
    'task': {},
    'subtask': {},
    'model': {},
}
p = Path('/data/hyogun/repos/haawron_mmaction2/work_dirs/train_output/cdar')
records = []
for p_pkl in p.rglob('best_pred.pkl'):
    task, subtask, model, add_on, extra_setting, jobname, job_array_idx, *_ = p_pkl.parts[p_pkl.parts.index('cdar')+1:]
    order, task = task.split('_', 1)
    orders['task'][task] = int(order)
    order, subtask = subtask.split('_', 1)
    orders['subtask'][subtask] = int(order)
    order, model = model.split('_', 1)
    orders['model'][model] = int(order)
    settings = {
        'task': task,
        'subtask': subtask,
        'model': model,
        'add_on': add_on,
        'extra_setting': extra_setting
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

if latex_view:
    sort_by_metrics = ['gcd_v2', 'gcd_v2_old', 'gcd_v2_new']
    df = df.sort_values(
        by=list(map(lambda s: f'{s}_order', sort_bys)) + sort_by_metrics,
        ascending=[True]*len(sort_bys)+[False]*len(sort_by_metrics)
    )
    df = df.groupby(['task', 'subtask', 'model']).head(1)
    df = df[sort_bys + sort_by_metrics]
    # df = df.set_index('model')
    for unique_task in df.task.unique():
        df_ = df[df.task == unique_task]
        del df_['task']
        # df_ = df_.
        with pd.option_context('display.precision', 1):
            print(df_)

else:
    sort_by_metrics = [
        'gcd_v2_balanced', 'gcd_v2_balanced_old', 'gcd_v2_balanced_new',
        'gcd_v2', 'gcd_v2_old', 'gcd_v2_new',
    ]
    df = df.set_index(['task', 'subtask', 'model', 'add_on', 'extra_setting'])
    df = df.sort_values(
        by=list(map(lambda s: f'{s}_order', sort_bys)) + sort_by_metrics + ['jid'],
        ascending=[True]*len(sort_bys)+[False]*len(sort_by_metrics)+[True]).fillna('-')

    for sort_by in sort_bys:
        del df[f'{sort_by}_order']

    cols = [
        'gcd_v2_balanced', 'gcd_v2_balanced_old', 'gcd_v2_balanced_new',
        'gcd_v2', 'gcd_v2_old', 'gcd_v2_new',
        'kmeans', 'kmeans_old', 'kmeans_new',
        'jid'
    ]
    df = df[cols]
    with pd.option_context('display.precision', 1):
        print(df)
