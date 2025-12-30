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
    dirlist=$(find . -type l | sed 's/^\.\///')  # liste des liens relatifs
    full_list=$(echo "$dirlist" | sed "s|^|${s3_storage}/${picklefiles_dir}/logs/|")  # ajouter le chemin de base
    mc mb ${full_list}
    popd

    count=0
    for dd in ${dirlist}
    do
        ((count++))
        if (( count % 9 == 0 )); then
            sleep 5
        else
            sleep 0.5
        fi
        mc cp -r ${picklefiles_path}/logs/${dd}/ ${s3_storage}/${picklefiles_dir}/logs/${dd}/
    done
    echo "Logs pushed to: ${picklefiles_path}/logs/"

    mc cp ${picklefiles_path}/*-best.* ${s3_storage}/${picklefiles_dir}/
    res=$?
    [ ${res} -ne 0 ] && echo "Error ${res}. S3, copy best files" >> /dev/stderr && exit 2
    bakdir=${picklefiles_path}/.bak$$
    mkdir -p ${bakdir}
    mv ${picklefiles_path}/*-best.pickle ${bakdir}/.

    # Nove files to S3. Do not keep locally all models to save disk (important in some )
    mc mv ${picklefiles_path}/*.pickle ${s3_storage}/${picklefiles_dir}_disq/
    res=$?
    mv ${bakdir}/*-best.pickle ${picklefiles_path}/.
    rmdir ${bakdir}
    [ ${res} -ne 0 ] && echo "Error ${res}. S3, move disqualified pickles to" >> /dev/stderr && exit 3 
else
    echo "S3 storage location Not Known! Results not saved."
fi

exit 0
