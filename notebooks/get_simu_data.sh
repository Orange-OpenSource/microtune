#!/bin/bash
# Usage: bash get_simu_data.sh
# Script to download all logs from S3 for different versions, datasets and models,
# and concatenate specific log files into a single log file for analysis.

version="13sgu"
version_minor_list="11 12"
ds_list="orig dbsas cvae tabddpm"
model_list="ppo a2c dqn"

# Log filename to concatenate in a unque file
logname="run_all_agents_test"

logdir="/tmp/logs$$"
mkdir -p ${logdir}

for version_minor in ${version_minor_list}
do
    for ds in ${ds_list}
    do
        for model in ${model_list}
        do
            cmd="mc cp -r poc/s3selfcare-vstune/E${version}_${version_minor}_${ds}-${model}-training/logs/ ${logdir}/${ds}-${model}/"
            echo $cmd
            $cmd
        done
    done
done

version_minor_list4f=$(echo ${version_minor_list} | tr ' ' '-')
logfile="${version}_${version_minor_list4f}_${logname}.log"
echo "Concatenate all ${logname}.log files into ${logfile}..."
find ${logdir} -name "${logname}.log" -exec echo {}  \; | xargs cat >> ${logfile}
echo 'Lines count: '`wc -l ${logfile}`

echo "Cleaning up temporary log directory..."
rm -rf ${logdir}

exit $?