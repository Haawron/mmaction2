from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description='To Select the Best Model')
parser.add_argument('-j', '--job-id', type=int,
                    help='The main job id of the task')
args = parser.parse_args()

p_workdirs = Path(r'work_dirs')
p_selected_workdirs = [p_workdir for p_workdir in p_workdirs.glob('*') if str(args.job_id) in p_workdir.name]
p_bests = []
for p_array_workdir in p_selected_workdirs:
    p_best, = [p_pth for p_pth in p_array_workdir.glob('**/*.pth') if 'top1' in p_pth.name]
    p_bests.append(p_best)
print(p_bests)
# source, target = [word for word in str(p_best).replace('/', '_').split('_') if 'P' in word]
# print(p_best, source, target)
