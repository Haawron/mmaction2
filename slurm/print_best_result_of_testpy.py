from pathlib import Path
import argparse
import csv
import numpy as np
import pandas as pd

pd.options.display.float_format = '{:.1f}'.format
dataset = 'ek100'
model = 'dann'
openness = 'closed'
pk = 'top1_acc'
p_models = Path(f'work_dirs/test_output/{dataset}/{model}/{openness}')
p_sessions = [p for p in p_models.glob('*') if p.is_dir()]
best_models = []
for p_sess in p_sessions:  # ex) p_sess = {rootdir}/P02_04
    df_sess = pd.DataFrame()
    sess_name = p_sess.stem
    for p_csv in p_sess.glob('**/*.csv'):
        df = pd.read_csv(p_csv)
        df['jid'] = p_csv.stem
        df = df.set_index('jid')
        df_sess = df.join(df_sess)
    best_jid = df_sess.idxmax()[pk]
    best_model = df_sess.loc[best_jid]
    best_model *= 100  # for print in percents
    best_model['jid'] = best_jid
    best_model = best_model.rename(sess_name)
    best_models.append(best_model)

print(dataset, model, openness)
print(f'Selected the models by {pk}\n')
print(pd.concat(best_models, axis=1).T.sort_index())

# todos:
    # latex 형식으로 출력: open도 추가해야 함
    # 표 형식으로 출력
