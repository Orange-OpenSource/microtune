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

# IMPORTANT!! Depending on type of experimentations:
# - either generate pickles Datasets for ORIG and all known SIMU (DBSAS, VCAE and even more later) if not already done
#   Thus allowing to use exactly the same Test and Eval datasets for ALL experiments
# - or generate only train, eval, test datasets cache files from an already existing dataset (full) pickle file
python run_gen_pickles.py ${EXPE_GEN_PICKLES} $*
if [ $? -ne 0 ]; then
    echo "ERROR: run_gen_pickles.py failed!"
    exit 1
fi

# Push run_gen_pickles logs to S3 ?
mc -v &>> /dev/null
if [ $? -eq 0 ]
then
    run_gen_pickles_logs="/tmp/run_gen_pickles${pid}.log"
    python run_gen_pickles.py ${EXPE_GEN_PICKLES} $* --cfg job --resolve > ${run_gen_pickles_logs}  2>&1
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_gen_pickles Configuration List" >> /dev/stderr && exit 100 

    s3_storage="$(grep s3_storage ${run_gen_pickles_logs} | cut -d' ' -f 2)"
    link2logs="$(grep link2logs ${run_gen_pickles_logs} | cut -d' ' -f 2)"
    logs_dirname_path="$(dirname ${link2logs})"
    logs_dirname="$(grep pickles_dirname ${run_gen_pickles_logs} | cut -d' ' -f 2)"
    mc mb -p -q "${s3_storage}/${logs_dirname}"
    cmd="mc cp -r ${logs_dirname_path}/  ${s3_storage}/${logs_dirname}/"
    echo "Save run_gen_pickles logs to S3: ${cmd}"
    ${cmd}    
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_gen_pickles" >> /dev/stderr 
fi

# Train and test experiments
./expetraintest.sh ${OPTUNA} ${expe_list} $*
./expetestdisqualified.sh ${expe_list} $*
./expesimpletest.sh ${expe_list} $*

exit $?


