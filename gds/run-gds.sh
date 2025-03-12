WORK_DIR=/opt/gds
GDS_VOLUME=/GDS_MOUNT
IMAGE="gds"

./gds-run-container run --rm --gpus=all --enable-mofed --enable-gds\
            --volume ${GDS_VOLUME}:/data:rw \
            --workdir ${WORK_DIR} \
            -it ${IMAGE} bash
