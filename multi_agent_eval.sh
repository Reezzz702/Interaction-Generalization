#!/bin/sh
export CARLA_ROOT=${1:-/home/hcis-s15/Documents/projects/RiskBench/CARLA_0.9.14_instance_id}
export WORKDIR=${2:-//home/hcis-s15/Documents/projects/Interaction_Genearlization}

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORKDIR}/team_code

export EGO_AGNET=${WORKDIR}/SRL_agent
export AGENT_CONFIG=${WORKDIR}/checkpoints/tfpp_wp
export DIRECT=0

killall -9 -r CarlaUE4-Linux 
bash ${CARLA_ROOT}/CarlaUE4.sh -quality-level=Low &

sleep 5

python multi_agent_eval.py \
--ego_agent=${EGO_AGNET} \
--agent_config=${AGENT_CONFIG}


# A0(error), A1, A6
# B3, B7, B8

# press g to 
