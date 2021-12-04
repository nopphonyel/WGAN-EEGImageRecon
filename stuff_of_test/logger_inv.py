import os
from libs.utils.logger import *

__dir_name__ = os.path.dirname(__file__)
LOG_FILE_ROOT = os.path.join(__dir_name__, "export_content/log/")


def j_p(p, *ps):
    return os.path.join(p, *ps)


l = LoggerGroup()
l.load_file(j_p(LOG_FILE_ROOT, "acc.lggr"))
l.plot_all()
