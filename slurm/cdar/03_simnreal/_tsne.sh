#!/bin/bash


if [ -z "$workdir" ]; then
    echo 'workdir not defined'
fi

jobname="$(basename "$(dirname "$(dirname "$workdir")")" | perl -pe 's/\d+__//g')"

python /data/gyeongho/framework/dummy/nc_CDAR/tsne.py \
    --feature-dir "$workdir" --title "$jobname"
