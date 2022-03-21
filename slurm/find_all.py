from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='To Select the Best Model')
parser.add_argument('-j', '--job-id', type=int,
                    help='The main job id of the task')
args = parser.parse_args()

p_workdirs = Path(r'work_dirs')
p_selected_workdirs = sorted([p_workdir for p_workdir in p_workdirs.glob('*') if str(args.job_id) in p_workdir.name])
p_pths = []
for p_array_workdir in p_selected_workdirs:
    p_pths_ = sorted([p_pth for p_pth in p_array_workdir.glob('**/*.pth') if 'epoch' in p_pth.name and 'best' not in p_pth.name])
    source, target = [part for part in str(p_pths_[0]).replace('/', '_').split('_') if 'P' in part]
    for i, p_pth in enumerate(p_pths_):
        p_pths.append((str(i), str(p_pth), source, target))

for line in p_pths:
    print(' '.join(line))
