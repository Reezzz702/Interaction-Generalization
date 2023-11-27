#!/bin/sh
killall -9 -r CarlaUE4-Linux 
../../CarlaUE4.sh -quality-level=Low &

sleep 5
# python manual_control_1.py --town Town05
python multi_agent_eval.py --town Town05 --mode auto #A1 #Town10HD #B8


# A0(error), A1, A6
# B3, B7, B8

# press g to 
