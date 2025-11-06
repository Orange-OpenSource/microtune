#!/bin/bash
# Usage: bash get_simu_data.sh
# Script to download all logs from S3 for different versions, datasets and models,
# and concatenate specific log files into a single log file for analysis.

version="13sgu"
version_minor_desc="11: orig+dbsas+cvae, 12:+tabddpm,+orig_with_all_features"
version_minor_list="11 12" 
ds_list="orig dbsas cvae tabddpm"
model_list="ppo a2c dqn"
expe_name_list="training training-allfeat"

logdir="/tmp/logs$$"
mkdir -p ${logdir}

for expe_name in ${expe_name_list}
do
    for version_minor in ${version_minor_list}
    do
        for ds in ${ds_list}
        do
            sleep 2 # Throttle to avoid overwhelming the S3 server
            for model in ${model_list}
            do
                typeset -i count=0
                typeset -i exit_code=1
                while [ $exit_code -ne 0 ] && [ $count -lt 3 ]
                do
                    cmd="mc cp -q -r poc/s3selfcare-vstune/E${version}_${version_minor}_${ds}-${model}-${expe_name}/logs/ ${logdir}/${ds}-${model}/"
                    echo $cmd
                    output=$($cmd 2>&1)
                    exit_code=$?

                    if [ $exit_code -ne 0 ]; then
                        # Vérifier le contenu de la sortie pour distinguer les erreurs
                        if echo "$output" | grep -q "Unable to prepare URL for copying. Object does not exist"; then
                            echo "Object does not exist. Continue..."
                            exit_code=0  # Ne pas retenter pour cette erreur
                        else
                            echo "$output"
                            count=$((count + 1))
                            echo "Retrying (${count})..."
                            sleep 2
                        fi
                    fi                    
                done
            done
        done
    done
done

# Log filename to concatenate in a unque file
logname="run_all_agents_test"

version_minor_list4f=$(echo ${version_minor_list} | tr ' ' '-')
logfile="E${version}_${version_minor_list4f}_${logname}.log"

echo "Version: ${version}, minor versions: ${version_minor_list}, datasets: ${ds_list}, models: ${model_list}, experiments: ${expe_name_list}" > ${logfile}
echo Description - ${version_minor_desc} >> ${logfile}

echo "Concatenate all ${logname}.log files into ${logfile}..."
find ${logdir} -name "${logname}.log" -exec echo {}  \; | xargs cat >> ${logfile}
echo 'Lines count: '`wc -l ${logfile}`

echo "Cleaning up temporary log directory..."
rm -rf ${logdir}

exit $?