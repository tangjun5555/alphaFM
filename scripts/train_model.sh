#!/bin/bash

# initial training
cat ${train_file} | ${alphaFM_bin}/fm_train \
-m ${model_first_version_file} -mf txt \
-core ${core_num} \
-dim 1,1,${vector_dim} \
-fvs 1 \
-mnt float

# incremental training
cat {train_file} | {alphaFM_bin}/fm_train \
-m ${model_current_version_file} -mf txt \
-im ${model_last_version_file} -imf txt \
-core ${core_num} \
-dim 1,1,${vector_dim} \
-fvs 1 \
-mnt float
