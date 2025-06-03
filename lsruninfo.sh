#!/bin/bash

[ ! -d "./$1" ] && echo "Invalid ./$1. Please specify output dir as parameter" >> /dev/stderr && exit 1
cd ./$1

lst=$(grep 'RunInfo:' *run_agent*/run_agent*.log | awk -F'RunInfo:' '{print $2}' | sort -u)

typeset -i len=0

B_IFS="${IFS}"
IFS=$'\n'
for ll in ${lst}
do
  echo ${ll}
  len=len+1
done
IFS="${B_IFS}"

echo "** $len results **"

exit $?
