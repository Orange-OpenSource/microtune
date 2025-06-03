#!/bin/bash

[ ! -d "./$1" ] && echo "Invalid ./$1. Please specify output dir as parameter" >> /dev/stderr && exit 1
cd ./$1

lst=$(grep 'Sweep with:' *run_agent*/run_agent*.log | awk -F'Sweep with:' '{print $2" "$1}' | sed 's/^[[:space:]]*(\([^,]*,[^)]*\))/\1/' | awk -F: '{print $1}' | sort -n | sed 's/ /!!!/g')

typeset -i len=0

for ll in ${lst}
do
  if [ $len == 0 ]
  then
    best=$(echo $ll | sed 's/!!!/ /g')
  fi

  echo ${ll} | sed 's/!!!/ /g'
  len=len+1
done

worst=$(echo $ll | sed 's/!!!/ /g')

echo "** $len results **"
echo "Best params at $best:"
cat ./$(echo $best | awk -F'/' '{print $1}' | awk '{print $NF}')/.hydra/overrides.yaml

echo "Worst params at $worst:"
cat ./$(echo $worst | awk -F'/' '{print $1}' | awk '{print $NF}')/.hydra/overrides.yaml

exit $?