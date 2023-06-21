import os

import argparse
from pathlib import Path
import re
from dateutil.parser import parse as date_parse
from datetime import timedelta
from typing import List, Dict, Any
from collections import defaultdict
from textwrap import indent
from sty import ef
import socket

import numpy as np
import pandas as pd


dispmap = {
    'top1_acc': 'Top1',
    'top5_acc': 'Top5',
    'mean_class_accuracy': 'MCA',
}


is_vosuda = lambda subtask: 'closed' in subtask or 'vosuda' in subtask or 'D' in subtask
print_args = [
    'display.precision', 1,
    'display.max_rows', None,
    'max_colwidth', None,
    'display.float_format', '{:.1f}'.format
]


def get_args():
    parser = argparse.ArgumentParser('VOSUDA_REPORT')
    parser.add_argument('-p', '--path', action='store_true')
    parser.add_argument('-v', '--valid', action='store_true', help='Show validation scores rather than test scores')
    parser.add_argument('-sh', '--show-hidden', action='store_true')
    parser.add_argument('-P', '--parsable', action='store_true', help='Print the tables in parsable mode')
    args = parser.parse_args()
    return args


class VOSUDAReport:
    def __init__(self,
            print_path:bool=False,
            valid_score_instead_of_test:bool=False,
            show_hidden=False,
            parsable=False,
    ):
        self.print_path = print_path
        self.valid_score_instead_of_test = valid_score_instead_of_test
        self.show_hidden = show_hidden
        username = 'gunsbrother' if any(
            [seraph_hostname in socket.gethostname() for seraph_hostname in ['ariel', 'moana', 'aurora']]
            ) else 'hyogun'
        self.p_root = Path(f"/data/{username}/repos/haawron_mmaction2/work_dirs/train_output/cdar/")
        self.p_valid_jobs = self.get_valid_job_dirs(self.p_root)
        job_dicts, orders = self.get_job_dicts_from_job_dirs(self.p_valid_jobs)

        df_jobs = pd.DataFrame.from_records(job_dicts)
        tasks = df_jobs['task'].unique()
        for i, task in enumerate(tasks):
            print(f'Task: {task}')
            print('\n')
            df_task = df_jobs[df_jobs['task']==task]
            subtasks = sorted(df_task['subtask'].unique(), key=orders['subtask'].get)
            for j, subtask in enumerate(subtasks):
                openness, dataset = subtask.split('_', 1)
                metric_name = 'top1_acc' if task in ['hello', 'ek100'] else 'mean_class_accuracy' if openness == 'closed' else 'H_mean_class_accuracy'
                print(f"Subtask: {openness.title()} {dataset.replace('_', ' → ')} ({metric_name})")
                print()
                df_subtask:pd.DataFrame = df_task[df_task['subtask']==subtask]
                df_subtask = df_subtask.sort_values(by='jid', kind='stable')
                # df_subtask = df_subtask.sort_values(by='job_array_idx')
                df_subtask = df_subtask.sort_values(by='model', key=lambda col: col.map(orders['model']), kind='stable')
                df_subtask = df_subtask.sort_values(by='backbone', key=lambda col: col.map(orders['backbone']), kind='stable')
                df_subtask = df_subtask.pivot_table(
                    values=metric_name,
                    index=['backbone', 'model', 'option1', 'option2', 'jid'],
                    columns=['job_array_idx'], sort=False
                )
                df_subtask = df_subtask.reindex(sorted(df_subtask.columns), axis=1)
                df_blank = pd.DataFrame(index=df_subtask.index)  # blank column for readability
                df_blank[' '] = ' '
                df_stats = pd.DataFrame(index=df_subtask.index)
                df_stats['best'] = df_subtask.max(axis=1)
                df_stats['avg'] = df_subtask.mean(axis=1)
                df_stats['std'] = df_subtask.std(axis=1)
                if parsable:
                    df_subtask = pd.concat([df_subtask, df_stats], axis=1)
                else:
                    df_subtask = pd.concat([df_blank, df_subtask, df_blank, df_stats], axis=1)
                jids = df_subtask.index.get_level_values('jid')
                if self.print_path:
                    df_paths = df_jobs.groupby('jid').head(1)[['path', 'jid']].set_index('jid').to_dict('series')['path']
                    df_paths = df_paths.apply(lambda p: p.relative_to(os.environ['PWD']).parent)  # 1. relative to PWD, 2. workdir for parent jid
                    df_subtask['p'] = jids.map(df_paths)
                df_subtask = df_subtask.rename(index={'default': '-'})
                with pd.option_context(*print_args):
                    msg = str(df_subtask.fillna('-'))
                    msg = indent(msg, prefix='    ')
                    msg = self.insert_seperator_line_when_options_change(msg, df_subtask)
                    msg = self.bold_average(msg)
                print(msg)
                if j < len(subtasks) - 1 or i < len(tasks) - 1:
                    print('\n')

            if i < len(tasks) - 1:
                print('-' * 18)

    def get_valid_job_dirs(self, p_root) -> List[Path]:
        '''
        valid 한지 아닌지 까보려면 p_log를 까봐야 되는데 그럴 거면 스코어 수집도 한번에 하는 게 낫지 않나?
        => 이렇게 나눠놔야 나중에 확장하기 편함
        '''
        def is_job_valid(p_job) -> bool:  # ex) p_job = 'work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/01_tsm/010_source_only/locality/l1_3/39164__closed_k2b-tsm-l1/0/20230504-025745'
            p_log = next(p_job.glob('*.log'))
            project, task, subtask, backbone, model, option1, option2, jobname, job_array_idx, _ = p_job.parts[p_job.parts.index('train_output')+1:]
            print('\r', Path(*p_job.parts[p_job.parts.index('haawron_mmaction2')+1:]), end='')
            # is vosuda
            if not is_vosuda(subtask):
                return False
            if not self.show_hidden:
                is_hidden = option2.startswith('_')
                if is_hidden:
                    return False
            # done testing ==> the training successfully ended
            with p_log.open() as f:
                log = f.read()
                matched = list(re.finditer('Testing results of the best checkpoint', log))
                if len(matched) == 0:
                    return False
            return True

        valid_job_dirs = []
        for p_best_pkl in p_root.rglob('best_pred.pkl'):
            p_job:Path = p_best_pkl.parent
            if not is_job_valid(p_job):
                continue
            valid_job_dirs.append(p_job)
        print('\r')
        return valid_job_dirs

    def get_job_dicts_from_job_dirs(self, p_valid_jobs:List[Path]):
        def parse_job_path(p_job:Path) -> List[dict]:
            project, task, subtask, backbone, model, option1, option2, jobname, job_array_idx, _ = p_job.parts[p_job.parts.index('train_output')+1:]
            order_task, task = task.split('_', 1)
            order_subtask, subtask = subtask.split('_', 1)
            order_model, model = model.split('_', 1)
            order_backbone, backbone = backbone.split('_', 1)
            jid = int(jobname.split('__')[0])
            job_array_idx = int(job_array_idx)
            job_dict = {
                'path': p_job,
                'project': project,
                'task': task,
                'subtask': subtask,
                'backbone': backbone,
                'model': model,
                'option1': option1,
                'option2': option2,
                'jid': jid,
                'job_array_idx': job_array_idx
            }
            order = {
                'task': {task: order_task},
                'subtask': {subtask: order_subtask},
                'model': {model: order_model},
                'backbone': {backbone: order_backbone},
            }
            return job_dict, order

        def get_job_score(p_log:Path):
            with p_log.open() as f:
                log = f.read()

                dates = re.findall(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}', log)  # dates written at the heads of logs, ex) 2023-04-17 06:12:35,440
                date_start, date_end = date_parse(dates[0]), date_parse(dates[-1])
                date_delta = date_end - date_start
                job_time_dict = {
                    'date_start': date_start,
                    'date_end': date_end,
                    'date_delta': date_delta
                }

                matched = list(re.finditer('Testing results of the best checkpoint', log))
                assert len(matched) > 0, f'something\'s got wrong with the validity check of this job {str(p_job)}'  # should have passed the validity check
                matched = matched[0]
                log_trunc = log[matched.end(0):]
                metric_dict:Dict[str,float] = {metric: 100*float(score) for metric, score in re.findall(r'(\w+): (\d+\.\d+)', log_trunc)}

                if self.valid_score_instead_of_test:
                    pattern = r'Best.*\n.*Epoch\(val\) \[(\d+)\]\[\d+\]\s+([\w\s:.,]+)\n'
                    epoch, metrics = re.findall(pattern, log)[-1]  # best valid
                    epoch = int(epoch)
                    valid_metric_dict:Dict[str,float] = {metric: 100*float(score) for metric, score in re.findall(r'(\w+): (\d\.\d{4})', metrics)}
                    metric_dict.update(valid_metric_dict)

            return job_time_dict, metric_dict

        job_dicts:List[Dict[str,Any]] = []
        order_dicts:List[Dict[str,Dict[str,Any]]] = []
        for p_job in p_valid_jobs:
            # obtain other information from the path itself
            job_dict, order_dict = parse_job_path(p_job)

            # obtain time info and metrics
            p_log = next(p_job.rglob('*.log'))
            job_time_dict, metric_dict = get_job_score(p_log)
            job_dict.update(job_time_dict)
            job_dict.update(metric_dict)

            job_dicts.append(job_dict)
            order_dicts.append(order_dict)

        # msg_time = f"{date_start.strftime(r'%Y-%m-%d %H:%M:%S')} ~ {date_end.strftime(r'%Y-%m-%d %H:%M:%S')}, took {timedelta(seconds=date_delta.seconds)}"
        # print(msg_time, job_dict, order_dict, metric_dict)

        orders:Dict[str,Dict[str,str]] = defaultdict(dict)
        for order_dict in order_dicts:
            for sort_by, child_order_dict in order_dict.items():
                for name, order in child_order_dict.items():
                    orders[sort_by][name] = order
        return job_dicts, orders

    def insert_seperator_line_when_options_change(self, msg, df_subtask):
        header_offset = 2  # the header takes 2 lines
        models = df_subtask.index.get_level_values(0)
        option1 = df_subtask.index.get_level_values(1)
        indices_models_change = [i+header_offset for i, (v1, v2) in enumerate(zip(models, models[1:])) if v1 != v2]  # sep line should be inserted after
        indices_option1_change = [i+header_offset for i, (v1, v2) in enumerate(zip(option1, option1[1:])) if v1 != v2]  # sep line should be inserted after
        indices_option1_change = sorted(set(indices_option1_change) - set(indices_models_change))
        indices_models_change.insert(0, header_offset-1)
        lines = msg.split('\n')
        line_width = max(map(len, lines))
        indent_widths = sorted(set(map(len, re.findall(r'^ *', msg, re.MULTILINE))))
        model_indent_width, option1_indent_width = indent_widths[0], indent_widths[1]
        for i in indices_models_change:
            lines[i] += '\n' + (' '*model_indent_width + '='*(line_width-model_indent_width))
        for i in indices_option1_change:
            lines[i] += '\n' + (' '*option1_indent_width + '-'*(line_width-option1_indent_width))
        msg = '\n'.join(lines)
        return msg

    def bold_average(self, msg):
        def repl(m):
            avg = m.group(0)
            return ef.bold + avg + ef.rs
        msg = re.sub(r'(\d+\.\d)(?=\s*\d+\.\d$)', repl, msg, flags=re.MULTILINE)
        return msg


if __name__ == '__main__':
    args = get_args()
    app = VOSUDAReport(
        print_path=args.path,
        valid_score_instead_of_test=args.valid,
        show_hidden=args.show_hidden,
    )
