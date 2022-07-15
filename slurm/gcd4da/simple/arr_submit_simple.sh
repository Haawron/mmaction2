#!/bin/zsh

shfile=${1:-'slurm/dann/dann_tsm_ek100.sh'}
begin=${2:-'now'}

model=${shfile##*/}
model=${model%%_*}

sources=(P02 P02)
targets=(P04 P22)

if [ -z $shfile ]; then
    echo 'Specify the shell file.'
else
    echo $shfile "begins" $begin

    for i in 0 1; do
        source="${sources[i]}"
        target="${targets[i]}"
        
        if [ -z $jid ]; then  # if $jid is not set
            jid=$(
                sbatch --job-name=${model}_tsm_${source}_${target} -p vll \
                --export=ALL,source=$source,target=$target \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                sbatch --job-name=${model}_tsm_${source}_${target} -p vll \
                --export=ALL,source=$source,target=$target \
                --dependency=afterany:$jid \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        fi
        echo $source '-->' $target 'jid:' $jid
    done
fi

