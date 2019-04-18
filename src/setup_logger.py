import sys
sys.path.append("../.")

import logging
from variables import Variables

v = Variables()

# Create log files
f = open(v.log_f, 'w')
f.close()

# Create loggers
logging.basicConfig(filename=v.log_f, filemode='w', level=logging.DEBUG)
logger = logging.getLogger('log')
