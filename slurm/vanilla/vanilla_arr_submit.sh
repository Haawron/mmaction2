#!/bin/zsh

shfile=${1:-'slurm/vanilla/vanilla_tsm_ek100.sh'}
begin=${2:-'now'}

model=vanilla

echo $shfile "begins" $begin

tasks=('source' 'target')
opennesses=('closed' 'open')
num_classeses=(5 6)
for domain in P02 P04 P22; do
    for i in 0 1; do
        task="${tasks[$i]}"
        openness="${opennesses[$i]}"
        num_classes="${num_classeses[$i]}"
        if [ -z $jid ]; then  # if $jid is not set
            jid=$(
                sbatch --job-name=${model}_tsm_${domain}_${task}-only -p vll \
                --export=ALL,domain=$domain,task=${task}-only,openness=$openness,num_classes=$num_classes \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                sbatch --job-name=${model}_tsm_${domain}_${task}-only -p vll \
                --export=ALL,domain=$domain,task=${task}-only,openness=$openness,num_classes=$num_classes \
                --dependency=afterany:$jid \
                --begin=$begin \
                $shfile | sed 's/[^0-9]*//g'
            )
        fi
        echo "Domain: $domain $task-only jid: $jid"
    done
done
