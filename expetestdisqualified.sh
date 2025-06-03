#!/bin/bash

if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

expe_list="$1"
shift

[ -z "${expe_list}" ] && echo "Error, no experiment specified as argument" >> /dev/stderr && exit 1
expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

typeset -i res=0

pid=$$
tmpcfglist=''

# Check names of experiments
echo "Check names of experiments and exit if error appears..."
for expe in ${expe_list}
do
    python run_all_agents_test.py +experiment=${expe} $* --cfg job --resolve > /tmp/${expe}${pid}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_all_agents_test" >> /dev/stderr && exit 100 
    cat /tmp/${expe}${pid}
    tmpcfglist="/tmp/${expe}${pid} ${tmpcfglist}" 
done

s3_storage="$(python run_agent${OPTUNA}.py --cfg job --resolve | grep s3_storage | cut -d' ' -f 2)"

for expe in ${expe_list}
do
    picklefiles_dir="$(grep pickles_dirname /tmp/${expe}${pid} | cut -d' ' -f 2)"
    picklefiles_disq_dir="$(grep pickles_disq_dirname /tmp/${expe}${pid} | cut -d' ' -f 2)"
    picklefiles_disq_path="$(grep pickles_disq_path /tmp/${expe}${pid} | cut -d' ' -f 2)"

    if [ ! -d ${picklefiles_disq_path} ] 
    then
        mkdir ${picklefiles_disq_path}
        cmd="mc cp -r ${s3_storage}/${picklefiles_disq_dir}/agent ${picklefiles_disq_path}/"
        echo "Retrieves models: ${cmd}"
        ${cmd}
    fi

    n_trials=$(ls -1 ${picklefiles_disq_path}/agent*.pickle | wc -l)
    cmd="python run_all_agents_test.py +experiment=${expe} ++n_trials=${n_trials} ++pickles_path=${picklefiles_disq_path} $*"
    echo "TEST: ${cmd}"
    ${cmd}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_all_agent_test" >> /dev/stderr && continue 

    # Copy result graph to S3 
    cmd="mc cp ${picklefiles_disq_path}/*-disq.html ${s3_storage}/${picklefiles_dir}/"
    echo "Save results to S3: ${cmd}"
    ${cmd}    
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. expetestdisqualified" >> /dev/stderr 

    # Copy logs to S3 
    cmd="mc cp -r ${picklefiles_disq_path}/logs ${s3_storage}/${picklefiles_dir}/"
    echo "Save logs to S3: ${cmd}"
    ${cmd}    
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. expetestdisqualified" >> /dev/stderr
done

for ff in ${tmpcfglist}
do
    rm -f $ff
done

[ ${res} -ne 0 ] && echo "Some Error(s) occured." >> /dev/stderr && exit 2 

exit 0
