#!/bin/bash

# Correct DEST_DIR path
DEST_DIR="/Users/bence/Desktop/total_perspective_vortex/total_perspective_vortex/data/"

# Check if S001 already exists
if [ -d "${DEST_DIR}/S001" ]; then
    echo "Data files already exist in ${DEST_DIR}. Skipping download and organization."
    exit 0
else
    echo "Data files not found in ${DEST_DIR}. Proceeding with download and organization."
fi

# Wait before downloading
sleep 5

for i in $(seq 1 60); do
    if [ $i -lt 10 ]; then
        folder="S00$i"
    else
        folder="S0$i"
    fi
    wget -r -N -c -np https://physionet.org/content/eegmmidb/1.0.0/${folder}/\#files-panel
    sleep 2
done


SOURCE_DIR="/Users/bence/Desktop/total_perspective_vortex/physionet.org/content/eegmmidb/1.0.0/"
DEST_DIR="/Users/bence/Desktop/total_perspective_vortex/data/"

cp -R ${SOURCE_DIR}/S* ${DEST_DIR}/

cd ${DEST_DIR}

# Remove unwanted directories/files
rm -rf 044
rm -f SHA256SUMS.txt
rm -f wfdbcal

curr_dir=$PWD
rm -rf "${curr_dir}/physionet.org

echo "Data organization complete."

