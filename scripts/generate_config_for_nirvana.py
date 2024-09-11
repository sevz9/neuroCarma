"""
Creates text for Singlt Option to Text Output Cube

"""

import sys
from typing import List

sys.path.append("./")

from utils import Config

conf = Config()

conf_params = conf.to_dict()


resulting_string_container: List[str] = []

for param_name in conf_params:
    
    resulting_string_container.append(f'{param_name}: ${{global.{param_name}!"{getattr(conf, param_name)}"}}')

resulting_string = "{}\n".format('\n'.join(resulting_string_container))

with open("parameters_default_values_generator.txt", "w") as f_write:
    f_write.write(resulting_string)