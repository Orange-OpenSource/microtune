#!/bin/bash

s3_storage="$1"
picklefiles_path="$2"
picklefiles_dir=$(basename "./${picklefiles_path}")

if [ "${s3_storage}" != "None" ]
then
    mc mb ${s3_storage}/${picklefiles_dir}_disq
    mc mb ${s3_storage}/${picklefiles_dir}/logs
    mc cp -r ${picklefiles_path}/logs/ ${s3_storage}/${picklefiles_dir}/logs/
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. S3, logs dir/subdir creation" >> /dev/stderr && exit 1

    echo "Copy symlinks in logs dir..."    
    pushd ${picklefiles_path}/logs
    dirlist=$(find . -type l | sed 's/^\.\///')
    for dd in ${dirlist}
    do
        mc mb ${s3_storage}/${picklefiles_dir}/logs/${dd}
    done
    popd
    sleep 4
    for dd in ${dirlist}
    do
        mc cp -r ${picklefiles_path}/logs/${dd}/ ${s3_storage}/${picklefiles_dir}/logs/${dd}/
    done
    echo "Logs pushed to: ${picklefiles_path}/logs/"

    mc cp ${picklefiles_path}/*-best.* ${s3_storage}/${picklefiles_dir}/
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. S3, copy best files" >> /dev/stderr && exit 2
    bakdir=${picklefiles_path}/.bak$$
    mkdir -p ${bakdir}
    mv ${picklefiles_path}/*-best.pickle ${bakdir}/.
    mc mv ${picklefiles_path}/*.pickle ${s3_storage}/${picklefiles_dir}_disq/
    res=$?
    mv ${bakdir}/*-best.pickle ${picklefiles_path}/.
    rmdir ${bakdir}
    [ ${res} -ne 0 ] && echo "Error ${res}. S3, move disqualified pickles to" >> /dev/stderr && exit 3 
else
    echo "S3 storage location Not Known! Results not saved."
fi

exit 0
