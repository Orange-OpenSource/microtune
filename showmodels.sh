#!/bin/bash
# 

modelfile_list="E13sg_8_cvae-dqn-training_"

for modelfile in $modelfile_list
do
  mc cp poc/s3seflcare-vstune/${modelfile}
done

exit 0
