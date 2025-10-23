#!/bin/bash
# Run DBSAS experiments (with DBSAS, CVAE, ... simulators) and generates learning, eval, tests as well as graphs graphs.
# Store all on S3 if defined.


if [ "x$1" == "x--sleep" ]; then
    shift
    echo "Wait $1 seconds before starting..."
    sleep $1
    shift
fi

OPTUNA=""
if [ "x$1" == "x--optuna" ]; then
    OPTUNA="--optuna"
    shift
fi

EXPE_GEN_PICKLES=""
if [ "x$1" == "x--expedataset" ]; then
    shift
    EXPE_GEN_PICKLES="+experiment=$1"
    shift
fi

expe_list="$1"
shift

# IMPORTANT!! Generate pickles Datasets for ORIG and all known SIMU (DBSAS, VCAE and even more later) if not already done
# This allows to use the same Test and Eval datasets for ALL experiments
python run_simu_gen_pickles.py ${EXPE_GEN_PICKLES}

[ -z "${expe_list}" ] && expe_list="dbsas-ppo-training,dbsas-a2c-training,dbsas-dqn-training,cvae-ppo-training,cvae-a2c-training,cvae-dqn-training,orig-ppo-training,orig-a2c-training,orig-dqn-training"
#expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

#orig-a2c-training,orig-dqn-training,orig-ppo-training,dbsas-a2c-training,dbsas-dqn-training,dbsas-ppo-training
./expetraintest.sh ${OPTUNA} ${expe_list} $*
./expetestdisqualified.sh ${expe_list} $*
./expesimpletest.sh ${expe_list} $*

exit $?


