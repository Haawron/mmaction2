from pathlib import Path
import argparse
import csv
import pandas as pd
import re


# python slurm/print_best_result_of_testpy.py -j {16279..16284}
# python slurm/print_best_result_of_testpy.py -j {16707..16712}
# python slurm/print_best_result_of_testpy.py -j {16748..16753}


# work_dirs/test_output/ek100/tsm/vanilla/P02/source-only/tested_on_P02/open/16279.csv
# work_dirs/test_output/ek100/tsm/dann/P02_P04/open/16279.csv
parser = argparse.ArgumentParser(description='Print the Best Model\'s Info, Given JID Or Config Vars.')
parser.add_argument('-j', '--jids', default=None, nargs='+',
                    help='The main job ids of the task')
parser.add_argument('-m', '--metric', default='mca', choices=['mca', 'acc', 'unk'],
                    help='')
args = parser.parse_args()

metric = {
    'mca': 'mean_class_accuracy',
    'acc': 'top1_acc',
    'unk': 'recall_unknown'
}[args.metric]

p_test_workdir = Path(r'work_dirs/test_output')
infos = []
for jid in args.jids:
    p_csvs = [p for p in p_test_workdir.glob('**/*.csv') if jid in str(p) and p.is_file()]
    for p_csv in p_csvs:
        if 'vanilla' in str(p_csv):
            pattern = r'/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/(?P<domain>[\w-]+)/(?P<task>[\w-]+)/tested_on_(?P<tested_on>[\w-]+)/(?P<openness>[\w-]+)/(?P<jid>\d+).csv'
        else:
            pattern = r'/(?P<dataset>[\w-]+)/(?P<backbone>[\w-]+)/(?P<model>[\w-]+)/(?P<task>[\w-]+)/(?P<openness>[\w-]+)/(?P<jid>\d+).csv'
        m = re.search(pattern, str(p_csv))
        info = {k: v for k, v in m.groupdict().items() if v is not None}
        with p_csv.open('r') as f:
            reader = csv.reader(f)
            header = next(reader)
            values = next(reader)
            info.update({k: 100*float(v) for k, v in zip(header, values)})
        infos.append(info)
df = pd.DataFrame(infos)

with pd.option_context('display.float_format', '{:.1f}'.format):
    if 'vanilla' in df['model'].values:
        mask = df['domain'] == df['tested_on']
        df_same = df.loc[mask, :]
        df_diff = df.loc[~mask, :]
        rearrange = lambda df: df.sort_values(by=['task', 'tested_on']).set_index(['task', 'domain', 'tested_on', 'openness']).unstack(level=[1, 2, 3])[metric]
        print(rearrange(df_same))
        print()
        print(rearrange(df_diff))
    else:
        key = lambda column: column.map(lambda value: ''.join(value.split('_')[::-1])) if column.name == 'task' else column
        rearrange = lambda df: df.sort_values(by=['task'], key=key).set_index(['model', 'task', 'openness']).unstack(level=[1, 2])[[metric]]
        print(rearrange(df))

# todos:
    # latex 형식으로 출력: open도 추가해야 함
    # 표 형식으로 출력
