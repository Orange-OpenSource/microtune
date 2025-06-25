#!/bin/bash

# This script is used to run the online version of microtune.
# It is expected to be run from the root directory of the microtune project.

MYSQL_DATABASE=${MYSQL_DATABASE:=adbms}
MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:=adbms}
MICROTUNE_SHOW_CONFIG=${MICROTUNE_SHOW_CONFIG:="--cfg job --resolve"}
MICROTUNE_ITERATIONS_COUNT=${MICROTUNE_ITERATIONS_COUNT:=100}
MICROTUNE_ITERATIONS_DELAY=${MICROTUNE_ITERATIONS_DELAY:=60}

echo "Running online microtune with ${MICROTUNE_ITERATIONS_COUNT} iterations."
echo "Using database schema: ${MYSQL_DATABASE}"

echo "Microtune is enabled with a delay before to start tuning: ${MICROTUNE_ITERATIONS_DELAY} seconds..."
sleep ${MICROTUNE_ITERATIONS_DELAY}



CMD="run_best_agent_live.py db=node db.password="${MYSQL_ROOT_PASSWORD}"  db.database="${MYSQL_DATABASE}" \
    tuner.TEST_EPISODES_COUNT=${MICROTUNE_ITERATIONS_COUNT} \
    db.warmup.on_start=${MICROTUNE_ITERATIONS_DELAY} \
    tuner.env.state_selector_test.buf_reset_policy=stay
"

if [ -n "${MICROTUNE_SHOW_CONFIG}" ]; then
    echo "Configuration for microtune:"
    python ${CMD} ${MICROTUNE_SHOW_CONFIG}
fi

echo "Running microtune..."
python ${CMD}

exit $?