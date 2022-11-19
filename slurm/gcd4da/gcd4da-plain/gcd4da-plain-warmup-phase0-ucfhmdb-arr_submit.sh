#!/bin/bash

shfile=${1:-'slurm/gcd4da/gcd4da-plain/templates/template-gcd4da-plain-warmup-phase0-ucfhmdb.sh'}

tasks=('ucf2hmdb' 'hmdb2ucf')
ckpts=(
    work_dirs/train_output/ucf2hmdb/svt/vanilla/source-only/13766__vanilla-svt-ucf2hmdb-source-only/0/20221009-152212/best_mean_class_accuracy_epoch_5.pth
    work_dirs/train_output/hmdb2ucf/svt/vanilla/source-only/13763__vanilla-svt-hmdb2ucf-source-only/0/20221009-145733/best_mean_class_accuracy_epoch_30.pth
)

if [ -z $shfile ]; then
    echo 'Specify the shell file.'
else
    echo $shfile

    for i in {0..1}; do
        task="${tasks[i]}"
        ckpt="${ckpts[i]}"
        argstring="-f $shfile --task $task --ckpt $ckpt"
        if [ -z $jid ]; then  # if $jid is not set
            jid=$(
                python slurm/utils/commons/render_template.py $argstring \
                | sbatch | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                python slurm/utils/commons/render_template.py $argstring \
                | sbatch --dependency=afterany:$jid | sed 's/[^0-9]*//g'
            )
        fi
        echo $task 'jid:' $jid
    done
fi
