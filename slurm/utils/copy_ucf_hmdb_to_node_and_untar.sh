#!/usr/bin/bash

for dataset in ucf101 hmdb51; do
  p_local="/local_datasets/${dataset}"
  if [ ! -d "${p_local}" ]; then
    tarfile="/data/hyogun/repos/haawron_mmaction2/data/tarfiles/${dataset}_rawframes.tar"
    if [ ! -f $tarfile ]; then
      echo "$tarfile does not exist."
      exit 1
    fi

    echo "mkdir ${p_local} ..."
    mkdir $p_local

    echo "cd to ${p_local} ..."
    cd $p_local

    echo "copying ${tarfile} ..."
    cp $tarfile .

    echo "untar ..."
    tar -xf "${dataset}_rawframes.tar"

    echo "cleaning ..."
    mv "data/dataset/${dataset}/rawframes" ./
    rm -rf data
    rm "${dataset}_rawframes.tar"

    echo -e "done.\n\n"
  fi
done
