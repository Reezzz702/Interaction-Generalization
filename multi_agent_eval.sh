#!/bin/sh
export CARLA_ROOT=${1:-/home/hcis-s15/Documents/projects/RiskBench/CARLA_0.9.14_instance_id}

killall -9 -r CarlaUE4-Linux 
bash ${CARLA_ROOT}/CarlaUE4.sh -quality-level=Low &

sleep 5

python multi_agent_eval.py  #A1 #Town10HD #B8


# A0(error), A1, A6
# B3, B7, B8

# press g to 
