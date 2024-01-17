#!/bin/sh
export CARLA_ROOT=${1:-/home/hcis-s15/Documents/projects/RiskBench/CARLA_0.9.14_instance_id}
export WORK_DIR=${2:-//home/hcis-s15/Documents/projects/Interaction_Genearlization}

export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${WORK_DIR}/team_code

export EVAL_CONFIG=${WORK_DIR}/eval_config/1to1/town04.json
export E2E_AGNET=${WORK_DIR}/SRL_agent
export AGENT_CONFIG=${WORK_DIR}/checkpoints/tfpp_wp
export DIRECT=0
export RESUME=0
export CHECKPOINT_ENDPOINT=${WORK_DIR}/results/1to1/Roach_TFPP/interaction.json
export SAVE_PATH=${WORK_DIR}/results/1to1/Roach_TFPP


killall -9 -r CarlaUE4-Linux 
bash ${CARLA_ROOT}/CarlaUE4.sh &

sleep 5

python multi_agent_eval.py \
--eval_config ${EVAL_CONFIG} \
--e2e_agent ${E2E_AGNET} \
--agent_config ${AGENT_CONFIG} \
--checkpoint ${CHECKPOINT_ENDPOINT} \
--resume ${RESUME}

sleep 5

killall -9 -r CarlaUE4-Linux 

sleep 1
