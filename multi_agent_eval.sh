#!/bin/sh
killall -9 -r CarlaUE4-Linux 
../../CarlaUE4.sh -quality-level=Low &

sleep 5

python multi_agent_eval.py  #A1 #Town10HD #B8


# A0(error), A1, A6
# B3, B7, B8

# press g to 
