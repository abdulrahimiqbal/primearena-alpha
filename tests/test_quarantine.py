import importlib, sys

FORBIDDEN = [
    "primearena.env",
    "primearena.residual_env",
    "primearena.baselines",
    "primearena.expert",
    "primearena.evaluate",
    "primearena.eval_safety",
    "primearena.arena",
    "primearena.mcts",
    "primearena.train",
    "primearena.replay",
    "primearena.distributed",
]


def test_primelead_does_not_import_rl_internals():
    for m in list(sys.modules):
        if m.startswith("primearena"):
            del sys.modules[m]
    importlib.import_module("primearena.primelead")
    loaded = [m for m in sys.modules if any(m.startswith(f) for f in FORBIDDEN)]
    assert loaded == [], f"primelead transitively imports frozen RL modules: {loaded}"
