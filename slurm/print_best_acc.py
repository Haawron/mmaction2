from pathlib import Path
import re
import argparse


def get_best_info(p_logs):
    def get_test_acc_from_logfile(p_log):
        with p_log.open('r') as f:
            data = f.read()
        pattern = r'Testing results of the best checkpoint\n.*top1_acc: ([\d\.]+)'
        test_acc, = list(map(float, re.findall(pattern, data)))
        return test_acc
    arg = max(range(len(p_logs)), key=lambda i: get_test_acc_from_logfile(p_logs[i]))
    p_log = p_logs[arg]
    source, target = re.findall(r'P\d+', str(p_log))
    return get_test_acc_from_logfile(p_logs[arg]), source, target


parser = argparse.ArgumentParser(description='To Select the Best Model')
parser.add_argument('-j', '--job-id', type=int,
                    help='The main job id of the task')
args = parser.parse_args()

p_workdirs = Path(r'work_dirs')
p_target_workdir, = [p_workdir for p_workdir in p_workdirs.glob('*') if str(args.job_id) in str(p_workdir)]
p_best_ckpts = [p_best_ckpt for p_best_ckpt in p_target_workdir.glob('**/*.pth') if 'best_top1_acc' in str(p_best_ckpt)]
p_logs = [next(p_best_ckpt.parent.glob('*.log')) for p_best_ckpt in p_best_ckpts]  # ckpts and logs are in the same order
acc, source, target = get_best_info(p_logs)
print(args.job_id, source, target, acc)
