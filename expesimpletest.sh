#!/bin/bash
# This script to launch a simple test on Test dataset, in particular shows in detail the SLA performance 
# using the last experimentation and the best model.
# Usage:
#   ./expesimpletest.sh [--sleep N] [--trialseed T,S] [experiment_list] [extra_options]
# where:
#   --sleep N               : wait N seconds before starting
#   --trialseed T,S        : specify Trial T and Seed S to use for the model selection (default: best model if T<0, whatever S)


if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

expe_list="$1"
shift

[ -z "${expe_list}" ] && expe_list="linucb_kfoofw"
expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

typeset -i TRIAL=-1
typeset -i SEEDID=0

if [ "x$1" == "x--trialseed" ]; then
    shift
    trialseed="$(echo ${1} | sed 's/,/ /g')"
    shift
    TRIAL=$(echo ${trialseed} | cut -d' ' -f 1)
    SEEDID=$(echo ${trialseed} | cut -d' ' -f 2)
fi

typeset -i res=0

# Check names of experiments
echo "Check names of experiments..."
for expe in ${expe_list}
do
    python run_simple_test.py +experiment=${expe} ++trial=${TRIAL} ++seed=${SEEDID} --cfg job --resolve
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_simple_test" >> /dev/stderr && exit 100 
done

s3_storage="$(python run_simple_test.py --cfg job --resolve | grep s3_storage | cut -d' ' -f 2)"

for expe in ${expe_list}
do
    #python run_dataset.py +experiment=${expe} 
    #res=$?
    #[ ${res} -ne 0 ] && echo "Error ${res}. run_simple_test" >> /dev/stderr && exit 10 

    picklefiles_dir="$(python run_simple_test.py +experiment=${expe} $* --cfg job --resolve | grep pickles_dirname | cut -d' ' -f 2)"
    picklefiles_path="$(python run_simple_test.py +experiment=${expe} $* --cfg job --resolve | grep pickles_path | cut -d' ' -f 2)"

    # Trial and SeedID identification from best agent file path ?
    if [ ${TRIAL} -lt 0 ] 
    then
        best_agent_path="$(ls -1 ${picklefiles_path}/agent*-best.pickle | head -1)"
        if [ "x${best_agent_path}" == "x" ]
        then
            echo "Error, no best agent found in ${picklefiles_path}" >> /dev/stderr
            res=1
            break
        fi
        best_agent_path="${best_agent_path##*/}"  # on travaille sur le nom du fichier seulement

        if [[ "$best_agent_path" =~ ^agent-T([0-9]|[1-9][0-9]|100)S([0-9]|[1-9][0-9]|100)-.*\.pickle$ ]]; then
            TRIAL_EXPE="${BASH_REMATCH[1]}"
            SEEDID_EXPE="${BASH_REMATCH[2]}"
            echo "Using Best Trial=${TRIAL_EXPE} and Seed=${SEEDID_EXPE}"
            #read -p "Press [Enter] to continue..."
        else
            echo "Error, best agent file name ${best_agent_path} not conform to agent-TxxSyy-*.pickle" >> /dev/stderr
            res=2
            break
       fi
       xtra_opts=""
    else
        TRIAL_EXPE=${TRIAL}
        SEEDID_EXPE=${SEEDID}
        picklefiles_path=${picklefiles_path}_disq
        xtra_opts="++pickles_path=${picklefiles_path}"
    fi

    cmd="python run_simple_test.py +experiment=${expe} ++trial=${TRIAL_EXPE} ++seed=${SEEDID_EXPE} ${xtra_opts} $*"
    echo ${cmd}
    ${cmd}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_simple_test" >> /dev/stderr && continue 

    # Send to S3?
    mc -v &>> /dev/null
    if [ $? -eq 0 ]
    then
        echo mc cp ${picklefiles_path}/*-sla_perf*best.html "${s3_storage}/${picklefiles_dir}/"
        mc cp ${picklefiles_path}/*-sla_perf*best.html "${s3_storage}/${picklefiles_dir}/"
        [ ${res} -ne 0 ] &&  break 
    else
        echo "SLA Graph Results in ${picklefiles_path}, Not Saved to S3!"
    fi
done

[ ${res} -ne 0 ] && echo "Some Error(s) occured." >> /dev/stderr && exit 2 

exit 0


python run_simple_test.py +experiment=xx ++trial=0 ++seed=0
