#!/bin/bash

OPTUNA=""
if [ "x$1" == "x--optuna" ]; then
    OPTUNA="_optuna"
    shift
fi

tuners_list="$1"
shift

[ -z "${tuners_list}" ] && tuners_list="linucb_kfoofw"
tuners_list="$(echo ${tuners_list} | sed 's/,/ /g')"

# Calculer la version "mineure" en fonction du nombre de secondes écoulées sur les dernières 24 heures et le numéro de jour du mois courant
current_time=$(date +%s)
ver="fast$(date +%d)$((current_time % 86400))" #${RANDOM}"

A_OPTS="n_seeds=2 \
tuner.TRAINING_COVERAGE=1. \
version_minor=${ver}"

B_OPTS=""

typeset -i res=0

for tuner in ${tuners_list}
do
    python run_agent${OPTUNA}.py tuner=${tuner} ${A_OPTS} $* 
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_agent" >> /dev/stderr && exit 1 

    python run_best_agent.py tuner=${tuner} ${A_OPTS} ${B_OPTS}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_best_agent" >> /dev/stderr && exit 2 

    python run_best_agent_test.py -m tuner=${tuner} ${A_OPTS} $* ${B_OPTS}
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. run_best_agent_test" >> /dev/stderr && exit 3 
done

[ ${res} -ne 0 ] && echo "Some Error(s) occured." >> /dev/stderr && exit 2 

exit 0
