import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/mnt/c/Users/drako/university/robotic_systems_1/project_new/project3_robotics/src/install/myarm_node'
