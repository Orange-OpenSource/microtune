#!/bin/bash

ds_list="orig dbsas cvae tabddpm"
model_list="ppo a2c dqn"
version="13sgu"
version_minor="11"
logname="run_all_agents_test"

logdir="/tmp/logs$$"
mkdir -p ${logdir}

for ds in ${ds_list}
do
    for model in ${model_list}
    do
        cmd="mc cp -r poc/s3selfcare-vstune/E${version}_${version_minor}_${ds}-${model}-training/logs/ ${logdir}/"
        echo $cmd
        $cmd
    done
done

find ${logdir} -name "${logname}.log" -exec echo {}  \; | xargs cat >> ${version}_${version_minor}_${logname}.log

echo "Cleaning up temporary log directory..."
rm -rf ${logdir}

exit $?