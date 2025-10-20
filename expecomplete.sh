#!/bin/bash
# Run DBSAS experiments and generates learning, eval, tests as well as graphs graphs.
# Store all on S3 if defined.


if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

expe_list="$1"
shift

[ -z "${expe_list}" ] && expe_list="01-opt_a2c_alphabeta_sigmoid,01-opt_ppo_alphabeta_sigmoid"
#expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

#orig-a2c-training,orig-dqn-training,orig-ppo-training,dbsas-a2c-training,dbsas-dqn-training,dbsas-ppo-training
./expetraintest.sh --optuna ${expe_list}
./expetestdisqualified.sh ${expe_list}
./expesimpletest.sh ${expe_list} 

exit $?


