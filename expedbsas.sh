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

# Generate pickles Datasets for ORIG and DBSAS if not already done
python run_dbsas_gen_pickles.py +experiment=dbsas-gen-pickles

[ -z "${expe_list}" ] && expe_list="orig-ppo-training,dbsas-ppo-training,orig-a2c-training,dbsas-a2c-training"
#expe_list="$(echo ${expe_list} | sed 's/,/ /g')"

./expetraintest.sh ${expe_list}
./expetestdisqualified.sh ${expe_list}
./expesimpletest.sh ${expe_list} 

exit $?


