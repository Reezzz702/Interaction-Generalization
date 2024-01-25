import json
from copy import deepcopy

agents = ["auto", "plant", "roach", "sensor"]
eavl_agent = "auto"

f = open("1to1/eval_easy.json")
eval_config = json.load(f)
scenario_list = []
index = 0
for scenario in eval_config['available_scenarios']:
  for agent in agents:
    scenario['ego_agent']['type'] = eavl_agent
    scenario['Index'] = index
    index += 1
    for other_agent in scenario['other_agents']:      
      other_agent['type'] = agent
    temp_scenario = deepcopy(scenario)
    scenario_list.append(temp_scenario)
    if agent == eavl_agent:
      continue
    else:
      scenario['ego_agent']['type'] = agent
      scenario['Index'] = index
      index += 1
      for other_agent in scenario['other_agents']:      
        other_agent['type'] = eavl_agent
      temp_scenario = deepcopy(scenario)
      scenario_list.append(temp_scenario)
    
eval_config['available_scenarios'] = scenario_list
with open("1to1/auto_eval_easy.json", 'w') as fd:
  json.dump(eval_config, fd, indent=2, sort_keys=False)