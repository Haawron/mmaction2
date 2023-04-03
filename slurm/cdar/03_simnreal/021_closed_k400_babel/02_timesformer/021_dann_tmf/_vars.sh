#!/bin/bash

export project='cdar'
export task='03_simnreal'  # table name
export subtask='021_closed_k400_babel'  # column
export backbone='02_timesformer'
export model='021_dann_tmf'  # row

export config='configs/recognition/cdar/03_simnreal/021_closed_k400_babel/02_timesformer/021_dann_tmf/k2b_tsf_dann_tmf.py'
export ckpt='work_dirs/train_output/cdar/03_simnreal/021_closed_k400_babel/02_timesformer/010_source_only/default/randaug/34826__k2b-tsf-randaug/2/20230401-144129/best_mean_class_accuracy_epoch_45.pth'
