#!/bin/zsh

shfile=${1:-'slurm/osbp/templates/template-osbp-svt-ek100.sh'}

if [ -z $shfile ]; then
    echo 'Specify the shell file.'
else
    echo $shfile

    for target in P02 P04 P22; do
        for source in P02 P04 P22; do
            if [[ $source = $target ]]; then
                continue
            fi
            if [ -z $jid ]; then  # if $jid is not set
                jid=$(
                    python slurm/utils/commons/render_template.py -f $shfile --source $source --target $target \
                    | sbatch | sed 's/[^0-9]*//g'
                )
            else
                jid=$(
                    python slurm/utils/commons/render_template.py -f $shfile --source $source --target $target \
                    | sbatch --dependency=afterany:$jid | sed 's/[^0-9]*//g'
                )
            fi
            echo $source '-->' $target 'jid:' $jid
        done
    done
fi
