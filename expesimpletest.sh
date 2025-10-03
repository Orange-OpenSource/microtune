#!/bin/bash
# This script to launch the SLA performance on test(or eval) datasets using the last experimentation and the best model.


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

TRIAL=${1:-0}; shift
SEEDID=${1:-0}; shift

typeset -i res=0

# Check names of experiments
echo "Check names of experiments and exit if error appears..."
for expe in ${expe_list}
do
    python run_simple_test.py +experiment=${expe} --cfg job
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
    
    cmd="python run_simple_test.py +experiment=${expe} ++trial=${TRIAL} ++seed=${SEEDID} $*"
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
