#!/bin/bash

# bash slurm/vanilla/vanilla-ek100-arr_submit.sh slurm/vanilla/vanilla-svt-ek100.sh

shfile=${1:-'slurm/vanilla/vanilla_tsm_ek100.sh'}
begin=${2:-'now'}

model=vanilla
backbone=$([[ $shfile =~ vanilla[-_]([a-z]*)[-_] ]] && echo ${BASH_REMATCH[1]})

echo $shfile "begins" $begin

tasks=('source-only' 'target-only')
opennesses=('closed' 'open')
num_classeses=(5 6)
for domain in P02 P04 P22; do
    for i in 0 1; do
        task="${tasks[$i]}"
        openness="${opennesses[$i]}"
        num_classes="${num_classeses[$i]}"
        jobname=${model}-${backbone}-${domain}_${task}
        if [ -z $jid ]; then  # if jid is not given and this is the head job
            jid=$(
                sbatch --job-name=${jobname} \
                --export=ALL,domain=${domain},task=${task},openness=${openness},num_classes=${num_classes} \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                sbatch --job-name=${jobname} \
                --export=ALL,domain=${domain},task=${task},openness=${openness},num_classes=${num_classes} \
                --dependency=afterany:$jid \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        fi
        echo "$backbone $domain $task jid: $jid"
    done
done
