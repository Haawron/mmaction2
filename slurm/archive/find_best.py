from pathlib import Path
import re
import argparse

# finds the best(in the test-accuracy) model among the models trained with different parameters
#
# the target directory be like:
#
# work_dirs/
#   $jid_so_on/
#       0/...
#           ${some_name0}.log
#           best_top1_acc_epoch_${some_epoch0}.pth
#       1/...
#           ${some_name1}.log
#           best_top1_acc_epoch_${some_epoch1}.pth
#       ...
#
# Remark: dir 0, 1, ... are working dirs for similar models trained with different parameters
# ==> prints checkpint, source, target
#
# >>> python thisfile.py
# best_top1_acc_epoch_${some_epoch1}.pth P02 P04


def arg_best(p_logs):
    def get_test_acc_from_logfile(p_log):
        with p_log.open('r') as f:
            data = f.read()
        pattern = r'Testing results of the best checkpoint\n.*top1_acc: ([\d\.]+)'
        test_acc, = list(map(float, re.findall(pattern, data)))
        return test_acc
    return max(range(len(p_logs)), key=lambda i: get_test_acc_from_logfile(p_logs[i]))


parser = argparse.ArgumentParser(description='To Select the Best Model')
parser.add_argument('-j', '--job-id', type=int,
                    help='The main job id of the task')
args = parser.parse_args()

p_workdirs = Path(r'work_dirs')
p_target_workdir, = [p_workdir for p_workdir in p_workdirs.glob('*') if str(args.job_id) in str(p_workdir)]
p_best_ckpts = [p_best_ckpt for p_best_ckpt in p_target_workdir.glob('**/*.pth') if 'best_top1_acc' in str(p_best_ckpt)]
p_logs = [next(p_best_ckpt.parent.glob('*.log')) for p_best_ckpt in p_best_ckpts]  # ckpts and logs are in the same order
best_arg = arg_best(p_logs)

best_ckpt = p_best_ckpts[best_arg]
source, target = re.findall(r'P\d+', str(best_ckpt))
print(best_ckpt, source, target)
