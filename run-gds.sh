WORK_DIR=/usr/local/cuda-12.6/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="sentry"

./gds-run-container run --rm --gpus=all --enable-mofed --enable-gds\
                        CUFILE_ENV_PATH_JSON=/usr/local/cuda-12.6/gds/cufile.json \
                        --volume ${GDS_VOLUME}:/data:rw \
                        --workdir ${WORK_DIR} \
                        -it ${IMAGE} bash
