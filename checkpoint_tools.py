import json
import os

def create_default_json_msg():
		msg = {
						"progress": [],
						"records": [],
						"global record": {}
					}
		return msg
	
def parse_checkpoint(path_to_checkpoint):
	if os.path.exists(path_to_checkpoint):
		with open(path_to_checkpoint) as fd:
			checkpoint = json.load(fd)
	else:
		checkpoint = create_default_json_msg()
	
	return checkpoint