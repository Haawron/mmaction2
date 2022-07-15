#!/bin/zsh

shfile=${1:-'slurm/dann/dann_tsm_ek100.sh'}
begin=${2:-'now'}

model=${shfile#*/}
model=${model%/*}

if [ -z $shfile ]; then
    echo 'Specify the shell file.'
else
    echo $shfile "begins" $begin

    for source in P02 P04 P22; do
        for target in P02 P04 P22; do
            if [[ $source = $target ]]; then
                continue
            fi
            if [ -z $jid ]; then  # if $jid is not set
                jid=$(
                    sbatch --job-name=${model}_tsm_${source}_${target} -p batch \
                    --export=ALL,source=$source,target=$target \
                    --begin=$begin \
                    $shfile | sed 's/[^0-9]*//g'
                )
            else
                jid=$(
                    sbatch --job-name=${model}_tsm_${source}_${target} -p batch \
                    --export=ALL,source=$source,target=$target \
                    --dependency=afterany:$jid \
                    --begin=$begin \
                    $shfile | sed 's/[^0-9]*//g'
                )
            fi
            echo $source '-->' $target 'jid:' $jid
        done
    done
fi

