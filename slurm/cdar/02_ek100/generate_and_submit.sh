#!/bin/bash

template_file=${1:-'slurm/cdar/02_ek100/02_timesformer/template/tsf-warmup.sh.j2'}

echo "$template_file $jid"

idx=0

for target in D1 D2 D3; do
    for source_ in D1 D2 D3; do
        if [ $source_ == $target ]; then
            continue
        fi

        printf -v idxx '%02d' $idx
        if [ -z "$jid" ]; then  # if jid is not given and this is the head job
            jid=$(
                python slurm/utils/commons/render_template.py -f "$template_file" \
                --idx "$idxx" --source $source_ --target $target \
                | sbatch | sed 's/[^0-9]*//g'
            )
        else
            jid=$(
                python slurm/utils/commons/render_template.py -f "$template_file" \
                --idx "$idxx" --source $source_ --target $target \
                | sbatch --dependency=afterany:"$jid" | sed 's/[^0-9]*//g'
            )
        fi
        printf "%s %s -> %s, jid: %d\n" "$idxx" $source_ $target "$jid"
        (( idx++ ))
    done
done
