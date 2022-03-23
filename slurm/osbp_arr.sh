#!/bin/zsh

begin=${1:-'now'}
echo "begins" $begin

for source in P02 P04 P22; do
    for target in P02 P04 P22; do
        if [[ $source = $target ]]; then
            continue
        fi
        if [ -z $jid ]; then  # if $jid is not set
            jid=$(
                sbatch --job-name=osbp_tsm_${source}_${target} \
                --export=ALL,source=$source,target=$target \
                --begin=$begin \
                slurm/osbp_tsm.sh | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                sbatch --job-name=osbp_tsm_${source}_${target} \
                --export=ALL,source=$source,target=$target \
                --dependency=afterany:$jid \
                --begin=$begin \
                slurm/osbp_tsm.sh | sed 's/[^0-9]*//g'
            )
        fi
        echo $source '-->' $target 'jid:' $jid
    done
done
