#!/bin/bash

# bash slurm/cdar/cdar-phase0-ek100-from-vanilla-sanity-arr_submit.sh slurm/cdar/cdar-phase0-tsm_ek100-from-vanilla-sanity.sh

shfile=${1:-'slurm/cdar/cdar-phase0-tsm_ek100-from-vanilla-sanity.sh'}
begin=${2:-'now'}

model=cdar
backbone=$([[ $shfile =~ cdar-phase0[-_]([a-z]*)[-_] ]] && echo ${BASH_REMATCH[1]})

echo $shfile "begins" $begin

for domain in P02 P04 P22; do
    jobname=${model}-phase0-${backbone}_${domain}-from-vanilla-sanity
    if [ -z $jid ]; then  # if jid is not given and this is the head job
        jid=$(
            sbatch --job-name=${jobname} \
            --export=ALL,domain=${domain} \
            --begin=$begin \
            $shfile | sed 's/[^0-9]*//g'
        )
    else
        jid=$(
            sbatch --job-name=${jobname} \
            --export=ALL,domain=${domain} \
            --dependency=afterany:$jid \
            --begin=$begin \
            $shfile | sed 's/[^0-9]*//g'
        )
    fi
    echo "$backbone $domain jid: $jid"
done
