import pathlib
import sys

src = pathlib.Path(__file__).parent
sys.path.append(str(src))
sys.path.append(str(src / "utils"))
sys.path.append(str(src / "megadetector"))
