from pathlib import Path

from lib.config import load_config

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_overrides_parse_types():
    cfg = load_config(DEFAULT, ["model.dtype=float32", "part3.n_images=10", "activations.layers=[8, 16]",
                                "paths.out_root=/tmp/x"])
    assert cfg.model.dtype == "float32"
    assert cfg.part3.n_images == 10
    assert cfg.activations.layers == [8, 16]
    assert cfg.run_dir == Path("/tmp/x") / "default"
