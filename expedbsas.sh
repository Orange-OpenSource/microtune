#!/bin/bash
# Run DBSAS experiments (with DBSAS, CVAE, ... simulators) and generates learning, eval, tests as well as graphs graphs.
# Store all on S3 if defined.


if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

expe_list="$1"
shift

# Generate pickles Datasets for ORIG and SIMU if not already done
#python run_dbsas_gen_pickles.py +experiment=dbsas-gen-pickles
#python run_dbsas_gen_pickles.py +experiment=cvae-gen-pickles

[ -z "${expe_list}" ] && expe_list="dbsas-ppo-training,dbsas-a2c-training,dbsas-dqn-training,cvae-ppo-training,cvae-a2c-training,cvae-dqn-training,orig-ppo-training,orig-a2c-training,orig-dqn-training"
#expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

#orig-a2c-training,orig-dqn-training,orig-ppo-training,dbsas-a2c-training,dbsas-dqn-training,dbsas-ppo-training
./expetraintest.sh ${expe_list}
./expetestdisqualified.sh ${expe_list}
./expesimpletest.sh ${expe_list} 

exit $?


