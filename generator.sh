#!/bin/bash


# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

MAPPER_CLI=$MAPPER_TOOL_DIR/mapper-interactive-cli.py
INTSTART=15
INTEND=45
INTSTEP=5

OVLSTART=15
OVLEND=45
OVLSTEP=5

EPSSTART=33
EPSEND=33
EPSSTEP=1

for eps in $(seq $EPSSTART $EPSSTEP $EPSEND); do
    clusterer=dbscan
    minpts=5
    filter=PC1

    MAPPER_SPEC=${INPUT_FILE}_${clusterer}_eps_${eps}_minpts_${minpts}_filter_${filter}

    GRAPH_LOCAL_DIR=$GRAPH_DIR/${MAPPER_SPEC}

    mkdir $GRAPH_LOCAL_DIR

    echo Start Running Epsilon ${eps}
    python3 $MAPPER_CLI ${INPUT_DIR}/${INPUT_FILE} \
    --intervals $INTSTART:$INTEND:$INTSTEP \
    --overlaps $OVLSTART:$OVLEND:$OVLSTEP \
    --clusterer ${clusterer} \
    --eps ${eps} \
    --min_samples ${minpts} \
    --filter ${filter} \
    -output $GRAPH_LOCAL_DIR
    # echo $LOG_DIR
    # python3 extract_graph_info.py $GRAPH_DIR $LOG_DIR $MAPPER_SPEC

done