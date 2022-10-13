import pathlib
import sys

utils = pathlib.Path(__file__).parent
if not str(utils) in sys.path:
    sys.path.append(str(utils))
