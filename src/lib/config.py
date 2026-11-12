"""YAML config loading with dotted command-line overrides.

Every script takes ``--config path.yaml`` and any number of ``--set key.sub=value``
overrides, so a run is fully described by the config file plus the command line.
"""
import argparse
import copy
from pathlib import Path

import yaml


class Config(dict):
    """A dict whose keys can also be read as attributes (cfg.model.name)."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def wrap(cls, obj):
        if isinstance(obj, dict):
            return cls({k: cls.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [cls.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        def unwrap(o):
            if isinstance(o, dict):
                return {k: unwrap(v) for k, v in o.items()}
            if isinstance(o, list):
                return [unwrap(v) for v in o]
            return o
        return unwrap(self)

    # Convenience paths shared by all stages.
    @property
    def run_dir(self) -> Path:
        return Path(self.paths.out_root) / self.run_name

    @property
    def data_dir(self) -> Path:
        return Path(self.paths.data_root)


def _apply_override(cfg: dict, assignment: str):
    if "=" not in assignment:
        raise ValueError(f"Override must look like key.sub=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    value = yaml.safe_load(raw)  # parses numbers, lists, null, true/false
    node = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_config(path, overrides=()) -> Config:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    for o in overrides:
        _apply_override(cfg, o)
    return Config.wrap(cfg)


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="KEY=VALUE", help="override a config value, e.g. --set model.dtype=float32")
    return p


def config_from_args(args) -> Config:
    cfg = load_config(args.config, args.overrides)
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    # Keep a copy of the exact config used next to the outputs.
    with open(cfg.run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
    return cfg
