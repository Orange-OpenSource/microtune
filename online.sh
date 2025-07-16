#!/bin/bash

# This script is used to run the online version of microtune.
# It is expected to be run from the root directory of the microtune project.
whoami
id
ls -la /
cat /etc/passwd
cat /etc/group
ls -la /home

MYSQL_DATABASE_HOST=${MYSQL_DATABASE_HOST:="localhost"}
MYSQL_DATABASE=${MYSQL_DATABASE:=adbms}
MYSQL_MICROTUNE_PASSWORD=${MYSQL_MICROTUNE_PASSWORD:=adbms}
MICROTUNE_VERBOSE=${MICROTUNE_VERBOSE:=0}
MICROTUNE_ITERATIONS_COUNT=${MICROTUNE_ITERATIONS_COUNT:=0}
MICROTUNE_ITERATIONS_DELAY=${MICROTUNE_ITERATIONS_DELAY:=60}
export=${MICROTUNE_HYDRA_CUSTOM_ARGS:=""}

echo "Using database host: ${MYSQL_DATABASE_HOST}"
echo "Using database schema: ${MYSQL_DATABASE}"
echo "Microtune is enabled with a delay before to start tuning: ${MICROTUNE_ITERATIONS_DELAY} seconds."
if [ -z "${MICROTUNE_ITERATIONS_COUNT}" ] || [ "${MICROTUNE_ITERATIONS_COUNT}" -eq 0 ]; then
    MICROTUNE_ITERATIONS_COUNT="null"
    echo "MICROTUNE_ITERATIONS_COUNT is not set or is 0. Run indefinitely."
    iterations="indefinitely"
else
    echo "MICROTUNE_ITERATIONS_COUNT is set to ${MICROTUNE_ITERATIONS_COUNT}."
    iterations="for ${MICROTUNE_ITERATIONS_COUNT} iterations"
fi

CMD="run_best_agent_live.py \
    verbosity=${MICROTUNE_VERBOSE} \
    db=node \
    db.host=${MYSQL_DATABASE_HOST} db.password="${MYSQL_MICROTUNE_PASSWORD}"  db.database="${MYSQL_DATABASE}" \
    db.warmups.on_start=${MICROTUNE_ITERATIONS_DELAY} \
    tuner.env.state_selector_test.buf_reset_policy=stay \
    tuner.TEST_STEPS_PER_EPISODE=${MICROTUNE_ITERATIONS_COUNT} \
    ${MICROTUNE_HYDRA_CUSTOM_ARGS} \
"

pip list
env |grep PYTHON

if [ "${MICROTUNE_VERBOSE}" -ne 0 ]; then
    echo "Hydra Configuration passed to microtune:"
    python ${CMD} --cfg job --resolve
    echo "python ${CMD}"
fi

echo "Running microtune ${iterations}..."
python ${CMD}

exit $?