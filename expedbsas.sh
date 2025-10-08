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

python run_dbsas_gen_pickles.py +experiment=dbsas-gen-pickles

[ -z "${expe_list}" ] && expe_list="orig-ppo-training,dbsas-ppo-training,orig-a2c-training,dbsas-a2c-training"
#expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

./expetraintest.sh ${expe_list}
./expetestdisqualified.sh ${expe_list}

exit $?

#### BACKUP - NOT USED ANYMORE ####
TRIAL=${1:-0}; shift
SEEDID=${1:-0}; shift

typeset -i res=0

for expe in ${expe_list}
do
    ./expetraintest.sh ${expe}
    [ ${res} -ne 0 ] && echo "Error ${res}. expetraintest" >> /dev/stderr && continue
    ./expetestdisqualified.sh ${expe}
    [ ${res} -ne 0 ] && echo "Error ${res}. expetestdisqualified" >> /dev/stderr && continue
#    ./expesimpletest.sh ${expe} ${TRIAL} ${SEEDID}
#    [ ${res} -ne 0 ] && echo "Error ${res}. expesimpletest" >> /dev/stderr && continue
done

[ ${res} -ne 0 ] && echo "Some Error(s) occured." >> /dev/stderr && exit 2 

exit 0
