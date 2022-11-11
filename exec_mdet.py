from omegaconf import OmegaConf

from src.runner import Runner
from src.utils.config import RootConfig
from src.utils.tag import SessionTag

cli = OmegaConf.from_cli()  # command line interface config
if cli.get("config_path"):
    cli_conf = OmegaConf.merge(OmegaConf.load(cli.config_path), OmegaConf.from_cli())
else:
    cli_conf = OmegaConf.merge(OmegaConf.load("config/mdet.yaml"), OmegaConf.from_cli())
schema = OmegaConf.structured(
    RootConfig(session_root=cli_conf.get("session_root"), output_dir=cli_conf.get("output_dir"))
)
# print(schema)
# schema = OmegaConf.load("config/mdet.yaml")
config = OmegaConf.merge(schema, cli_conf)

runner = Runner(config=config, session_tag=SessionTag.MDet)
runner.execute()
