#!/bin/bash

shfile=${1:-'slurm/vanilla/templates/template-vanilla-svt-ek100.sh'}

echo $shfile

tasks=('source-only' 'target-only')
opennesses=('closed' 'open')
num_classeses=(5 6)
for domain in P02 P04 P22; do
    for i in 0 1; do
        task="${tasks[$i]}"
        openness="${opennesses[$i]}"
        num_classes="${num_classeses[$i]}"

        if [ -z $jid ]; then  # if jid is not given and this is the head job
            jid=$(
                python slurm/utils/commons/render_template.py -f $shfile --t $task --d $domain --n $num_classes --o $openness \
                | sbatch | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                python slurm/utils/commons/render_template.py -f $shfile --t $task --d $domain --n $num_classes --o $openness \
                | sbatch --dependency=afterany:$jid | sed 's/[^0-9]*//g'
            )
        fi
        echo "$domain $task jid: $jid"
    done
done
