#!/bin/bash

# This script is used to run the online version of microtune.
# It is expected to be run from the root directory of the microtune project.

MICROTUNE_ITERATIONS_COUNT=${MICROTUNE_ITERATIONS_COUNT:=100}

echo "Running online microtune with ${MICROTUNE_ITERATIONS_COUNT} iterations."
echo "Using database: ${MYSQL_DATABASE}"

python run_best_agent_live.py db=node db.password="${MYSQL_ROOT_PASSWORD}" \
    db.database="${MYSQL_DATABASE}" \
    tuner.TEST_EPISODES_COUNT=${MICROTUNE_ITERATIONS_COUNT}

exit $?