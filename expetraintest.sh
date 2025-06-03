#!/bin/bash

if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

OPTUNA=""
if [ "x$1" == "x--optuna" ]; then
    OPTUNA="_optuna"
    shift
fi

expe_list="$1"
shift

[ -z "${expe_list}" ] && expe_list="linucb_kfoofw"
expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

typeset -i res=0

# Check names of experiments
echo "Check names of experiments and exit if error appears..."
for expe in ${expe_list}
do
    python run_agent${OPTUNA}.py +experiment=${expe} --cfg job
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_agent${OPTUNA}" >> /dev/stderr && exit 100 
done

s3_storage="$(python run_agent${OPTUNA}.py --cfg job --resolve | grep s3_storage | cut -d' ' -f 2)"

for expe in ${expe_list}
do
    #python run_dataset.py +experiment=${expe} 
    #res=$?
    #[ ${res} -ne 0 ] && echo "Error ${res}. run_agent${OPTUNA}" >> /dev/stderr && exit 10 

    picklefiles_dir="$(python run_agent${OPTUNA}.py +experiment=${expe} $* --cfg job --resolve | grep pickles_dirname | cut -d' ' -f 2)"
    picklefiles_path="$(python run_agent${OPTUNA}.py +experiment=${expe} $* --cfg job --resolve | grep pickles_path | cut -d' ' -f 2)"
    
    python run_agent${OPTUNA}.py +experiment=${expe} $*
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_agent${OPTUNA}" >> /dev/stderr && continue 

    #[ ! -d ${picklefiles_path} ] && echo "Error: Unable to access ${picklefiles_path}" >> /dev/stderr && continue

    python run_best_agent.py +experiment=${expe} 
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_best_agent" >> /dev/stderr && continue 

    python run_best_agent_test.py +experiment=${expe}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_best_agent_test" >> /dev/stderr && continue 

    # Send to S3?
    mc -v &>> /dev/null
    if [ $? -eq 0 ]
    then
        ./push2s3.sh "${s3_storage}" "${picklefiles_path}"
        [ ${res} -ne 0 ] &&  break 
    else
        echo "Results in ${picklefiles_path}, Not Saved to S3!"
    fi
done

[ ${res} -ne 0 ] && echo "Some Error(s) occured." >> /dev/stderr && exit 2 

exit 0
