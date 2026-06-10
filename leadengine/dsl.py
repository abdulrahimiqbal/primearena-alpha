from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .core import Statistic, Window


Q_VALUES = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 30)
LAG_VALUES = (1, 2, 3)
FFT_K_VALUES = tuple(range(1, 17))
FHIST_BINS = (8, 16)
MAX_DEPTH = 8
MAX_COMPLEXITY = 20


@dataclass(frozen=True)
class Type:
    name: str


W = Type("W")
SeqInt = Type("SeqInt")
SeqFloat = Type("SeqFloat")
SeqPair = Type("SeqPair")
Vec = Type("Vec")
Scalar = Type("Scalar")


@dataclass(frozen=True)
class Expr:
    op: str
    typ: Type
    cost: int
    args: tuple["Expr", ...] = ()
    params: tuple[Any, ...] = ()

    def complexity(self) -> float:
        return float(self.cost + sum(arg.complexity() for arg in self.args))

    def depth(self) -> int:
        return 1 + max((arg.depth() for arg in self.args), default=0)

    def describe(self) -> str:
        if self.op == "w":
            return "w"
        inner = [arg.describe() for arg in self.args] + [str(p) for p in self.params]
        return f"{self.op}({','.join(inner)})"

    def shape(self) -> str:
        if self.op == "w":
            return "w"
        inner = [arg.shape() for arg in self.args] + ["#" for _ in self.params]
        return f"{self.op}({','.join(inner)})"

    def eval(self, w: Window) -> Any:
        vals = [arg.eval(w) for arg in self.args]
        if self.op == "w":
            return w
        if self.op == "positions":
            win = vals[0]
            idx = np.flatnonzero(np.asarray(win.values) > 0)
            if "integer_values" in win.meta:
                return np.asarray(win.meta["integer_values"], dtype=np.int64)[idx]
            return int(win.start) + idx.astype(np.int64)
        if self.op == "gaps":
            seq = np.asarray(vals[0], dtype=np.int64)
            return np.diff(seq).astype(np.int64) if seq.size > 1 else np.asarray([], dtype=np.int64)
        if self.op == "mod":
            return (np.asarray(vals[0], dtype=np.int64) % int(self.params[0])).astype(np.int64)
        if self.op == "pairs":
            seq = np.asarray(vals[0], dtype=np.int64)
            lag = int(self.params[0])
            if seq.size <= lag:
                return np.empty((0, 2), dtype=np.int64)
            return np.stack([seq[:-lag], seq[lag:]], axis=1).astype(np.int64)
        if self.op == "hist":
            q = int(self.params[0])
            seq = np.asarray(vals[0], dtype=np.int64)
            counts = np.bincount(seq % q, minlength=q).astype(np.float32) if seq.size else np.zeros(q, dtype=np.float32)
            return counts / max(float(counts.sum()), 1.0)
        if self.op == "pair_hist":
            q = int(self.params[0])
            pairs = np.asarray(vals[0], dtype=np.int64).reshape(-1, 2)
            out = np.zeros(q * q, dtype=np.float32)
            if pairs.size == 0:
                return out
            ids = (pairs[:, 0] % q) * q + (pairs[:, 1] % q)
            counts = np.bincount(ids.astype(np.int64), minlength=q * q).astype(np.float32)
            return counts / max(float(counts.sum()), 1.0)
        if self.op == "fourier_power":
            indicator = np.asarray(vals[0].values, dtype=np.float64)
            if indicator.size == 0:
                return 0.0
            spectrum = np.fft.rfft(indicator)
            k = min(int(self.params[0]), spectrum.size - 1)
            return float(np.abs(spectrum[k]) ** 2 / max(indicator.size, 1))
        if self.op == "logweight":
            seq = np.asarray(vals[0], dtype=np.float64)
            return (1.0 / np.maximum(np.log(np.maximum(seq, 3.0)), 1.0)).astype(np.float32)
        if self.op == "ratios":
            source = vals[0].values if isinstance(vals[0], Window) else vals[0]
            seq = np.asarray(source, dtype=np.float64)
            if seq.size <= 1:
                return np.empty(0, dtype=np.float32)
            a = seq[:-1]
            b = seq[1:]
            return (np.minimum(a, b) / np.maximum(np.maximum(a, b), 1e-12)).astype(np.float32)
        if self.op == "fhist":
            bins = int(self.params[0])
            if isinstance(vals[0], Window):
                seq = np.asarray(vals[0].values, dtype=np.float64)
                if vals[0].meta.get("domain") == "ff_angles":
                    seq = seq / np.pi
            else:
                seq = np.asarray(vals[0], dtype=np.float64)
            counts, _ = np.histogram(seq, bins=bins, range=(0.0, 1.0))
            counts = counts.astype(np.float32)
            return counts / max(float(counts.sum()), 1.0)
        if self.op == "mean":
            seq = np.asarray(vals[0], dtype=np.float64)
            return float(np.mean(seq)) if seq.size else 0.0
        if self.op == "var":
            seq = np.asarray(vals[0], dtype=np.float64)
            return float(np.var(seq)) if seq.size else 0.0
        if self.op == "normalize":
            vec = np.asarray(vals[0], dtype=np.float32).reshape(-1)
            total = float(np.sum(np.abs(vec)))
            return vec / total if total > 1e-12 else vec
        if self.op == "concat":
            return np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in vals]).astype(np.float32)
        if self.op == "scalar_vec":
            return np.asarray([float(vals[0])], dtype=np.float32)
        raise ValueError(f"Unknown DSL op: {self.op}")


def _validate(expr: Expr) -> Expr:
    if expr.depth() > MAX_DEPTH:
        raise ValueError(f"Program exceeds max depth {MAX_DEPTH}: {expr.describe()}")
    if expr.complexity() > MAX_COMPLEXITY:
        raise ValueError(f"Program exceeds max complexity {MAX_COMPLEXITY}: {expr.describe()}")
    return expr


def w_expr() -> Expr:
    return Expr("w", W, 0)


def positions(x: Expr) -> Expr:
    if x.typ != W:
        raise TypeError("positions expects W")
    return Expr("positions", SeqInt, 1, (x,))


def gaps(x: Expr) -> Expr:
    if x.typ != SeqInt:
        raise TypeError("gaps expects SeqInt")
    return Expr("gaps", SeqInt, 1, (x,))


def mod(x: Expr, q: int) -> Expr:
    if x.typ != SeqInt or int(q) not in Q_VALUES:
        raise TypeError("mod expects SeqInt and allowed q")
    return Expr("mod", SeqInt, 1, (x,), (int(q),))


def pairs(x: Expr, lag: int) -> Expr:
    if x.typ != SeqInt or int(lag) not in LAG_VALUES:
        raise TypeError("pairs expects SeqInt and allowed lag")
    return Expr("pairs", SeqPair, 1, (x,), (int(lag),))


def hist(x: Expr, q: int) -> Expr:
    if x.typ != SeqInt or int(q) not in Q_VALUES:
        raise TypeError("hist expects SeqInt and allowed q")
    return Expr("hist", Vec, 2, (x,), (int(q),))


def pair_hist(x: Expr, q: int) -> Expr:
    if x.typ != SeqPair or int(q) not in Q_VALUES:
        raise TypeError("pair_hist expects SeqPair and allowed q")
    return Expr("pair_hist", Vec, 2, (x,), (int(q),))


def fourier_power(x: Expr, k: int) -> Expr:
    if x.typ != W or int(k) not in FFT_K_VALUES:
        raise TypeError("fourier_power expects W and k in 1..16")
    return Expr("fourier_power", Scalar, 2, (x,), (int(k),))


def logweight(x: Expr) -> Expr:
    if x.typ != SeqInt:
        raise TypeError("logweight expects SeqInt")
    return Expr("logweight", SeqFloat, 1, (x,))


def ratios(x: Expr) -> Expr:
    if x.typ not in (SeqFloat, W):
        raise TypeError("ratios expects SeqFloat or a spacing Window")
    return Expr("ratios", SeqFloat, 1, (x,))


def fhist(x: Expr, bins: int) -> Expr:
    if x.typ not in (SeqFloat, W) or int(bins) not in FHIST_BINS:
        raise TypeError("fhist expects SeqFloat or W and bins in {8,16}")
    return Expr("fhist", Vec, 2, (x,), (int(bins),))


def mean(x: Expr) -> Expr:
    if x.typ not in (SeqInt, SeqFloat):
        raise TypeError("mean expects SeqInt or SeqFloat")
    return Expr("mean", Scalar, 1, (x,))


def var(x: Expr) -> Expr:
    if x.typ not in (SeqInt, SeqFloat):
        raise TypeError("var expects SeqInt or SeqFloat")
    return Expr("var", Scalar, 1, (x,))


def normalize(x: Expr) -> Expr:
    if x.typ != Vec:
        raise TypeError("normalize expects Vec")
    return Expr("normalize", Vec, 1, (x,))


def concat(a: Expr, b: Expr) -> Expr:
    if a.typ != Vec or b.typ != Vec:
        raise TypeError("concat expects Vec, Vec")
    return Expr("concat", Vec, 1, (a, b))


def scalar_vec(x: Expr) -> Expr:
    if x.typ != Scalar:
        raise TypeError("scalar_vec expects Scalar")
    return Expr("scalar_vec", Vec, 1, (x,))


CONSTRUCTORS: dict[str, Callable[..., Expr]] = {
    "positions": positions,
    "gaps": gaps,
    "mod": mod,
    "pairs": pairs,
    "hist": hist,
    "pair_hist": pair_hist,
    "fourier_power": fourier_power,
    "logweight": logweight,
    "ratios": ratios,
    "fhist": fhist,
    "mean": mean,
    "var": var,
    "normalize": normalize,
    "concat": concat,
    "scalar_vec": scalar_vec,
}


@dataclass
class Program(Statistic):
    root: Expr
    ood_auc: float = 0.0
    promoted: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.describe()

    def __call__(self, w: Window | list[Window]) -> np.ndarray:
        if isinstance(w, list):
            return np.stack([self(one) for one in w]).astype(np.float32)
        out = self.root.eval(w)
        if self.root.typ == Scalar:
            return np.asarray([float(out)], dtype=np.float32)
        return np.asarray(out, dtype=np.float32).reshape(-1)

    def complexity(self) -> float:
        return self.root.complexity()

    def describe(self) -> str:
        return self.root.describe()

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Program) and self.root == other.root


def _split_args(s: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append(s[start:i])
            start = i + 1
    tail = s[start:]
    if tail:
        args.append(tail)
    return [a.strip() for a in args]


def _parse_expr(s: str) -> Expr:
    s = s.strip()
    if s == "w":
        return w_expr()
    if "(" not in s or not s.endswith(")"):
        raise ValueError(f"Invalid expression: {s}")
    op, rest = s.split("(", 1)
    op = op.strip()
    inner = rest[:-1]
    raw_args = _split_args(inner)
    if op not in CONSTRUCTORS:
        raise ValueError(f"Unknown primitive: {op}")

    if op in {"mod", "hist", "pair_hist", "fhist"}:
        if len(raw_args) != 2:
            raise ValueError(f"{op} expects 2 arguments")
        return CONSTRUCTORS[op](_parse_expr(raw_args[0]), int(raw_args[1]))
    if op in {"pairs", "fourier_power"}:
        if len(raw_args) != 2:
            raise ValueError(f"{op} expects 2 arguments")
        return CONSTRUCTORS[op](_parse_expr(raw_args[0]), int(raw_args[1]))
    if op == "concat":
        if len(raw_args) != 2:
            raise ValueError("concat expects 2 arguments")
        return concat(_parse_expr(raw_args[0]), _parse_expr(raw_args[1]))
    if len(raw_args) != 1:
        raise ValueError(f"{op} expects 1 argument")
    return CONSTRUCTORS[op](_parse_expr(raw_args[0]))


def parse_program(s: str) -> Program:
    root = _validate(_parse_expr(s))
    if root.typ not in (Vec, Scalar):
        raise TypeError("Program root must be Vec or Scalar")
    return Program(root)


def program_shape(program: Program) -> str:
    return program.root.shape()


def _expr_paths(expr: Expr) -> list[tuple[int, ...]]:
    out = [()]
    for i, arg in enumerate(expr.args):
        out.extend((i, *tail) for tail in _expr_paths(arg))
    return out


def _expr_at(expr: Expr, path: tuple[int, ...]) -> Expr:
    for idx in path:
        expr = expr.args[idx]
    return expr


def _replace_expr(expr: Expr, path: tuple[int, ...], new: Expr) -> Expr:
    if not path:
        return new
    i = path[0]
    args = list(expr.args)
    args[i] = _replace_expr(args[i], path[1:], new)
    return Expr(expr.op, expr.typ, expr.cost, tuple(args), expr.params)


def _random_expr(rng: np.random.Generator, target_type: Type, max_depth: int, max_complexity: float) -> Expr:
    if max_depth <= 1:
        if target_type == W:
            return w_expr()
        raise ValueError("no terminal for requested non-window type")

    def build() -> Expr:
        if target_type == W:
            return w_expr()
        if target_type == SeqInt:
            choices = ["positions", "gaps", "mod"]
            if max_depth <= 2:
                choices = ["positions"]
            choice = str(rng.choice(choices, p=[0.34, 0.28, 0.38] if len(choices) == 3 else None))
            if choice == "positions":
                return positions(w_expr())
            if choice == "gaps":
                return gaps(_random_expr(rng, SeqInt, max_depth - 1, max_complexity - 1))
            return mod(_random_expr(rng, SeqInt, max_depth - 1, max_complexity - 1), int(rng.choice(Q_VALUES)))
        if target_type == SeqPair:
            return pairs(_random_expr(rng, SeqInt, max_depth - 1, max_complexity - 1), int(rng.choice(LAG_VALUES)))
        if target_type == SeqFloat:
            if rng.random() < 0.45 or max_depth <= 2:
                return ratios(w_expr())
            if rng.random() < 0.65:
                return logweight(_random_expr(rng, SeqInt, max_depth - 1, max_complexity - 1))
            return ratios(_random_expr(rng, SeqFloat, max_depth - 1, max_complexity - 1))
        if target_type == Scalar:
            if rng.random() < 0.35 or max_depth <= 2:
                return fourier_power(w_expr(), int(rng.choice(FFT_K_VALUES)))
            source = _random_expr(rng, SeqFloat if rng.random() < 0.45 else SeqInt, max_depth - 1, max_complexity - 1)
            return mean(source) if rng.random() < 0.5 else var(source)
        if target_type == Vec:
            choices = ["hist", "pair_hist", "fhist", "normalize", "concat", "scalar_vec"]
            if max_depth <= 2:
                choices = ["hist", "fhist", "scalar_vec"]
            choice = str(rng.choice(choices, p=[0.23, 0.30, 0.18, 0.10, 0.09, 0.10] if len(choices) == 6 else None))
            if choice == "hist":
                q = int(rng.choice(Q_VALUES))
                return hist(_random_expr(rng, SeqInt, max_depth - 1, max_complexity - 2), q)
            if choice == "pair_hist":
                q = int(rng.choice(Q_VALUES))
                return pair_hist(_random_expr(rng, SeqPair, max_depth - 1, max_complexity - 2), q)
            if choice == "fhist":
                source = w_expr() if rng.random() < 0.45 else _random_expr(rng, SeqFloat, max_depth - 1, max_complexity - 2)
                return fhist(source, int(rng.choice(FHIST_BINS)))
            if choice == "normalize":
                return normalize(_random_expr(rng, Vec, max_depth - 1, max_complexity - 1))
            if choice == "concat":
                left_budget = max(1.0, (max_complexity - 1) * float(rng.uniform(0.35, 0.65)))
                right_budget = max(1.0, max_complexity - 1 - left_budget)
                return concat(
                    _random_expr(rng, Vec, max_depth - 1, left_budget),
                    _random_expr(rng, Vec, max_depth - 1, right_budget),
                )
            return scalar_vec(_random_expr(rng, Scalar, max_depth - 1, max_complexity - 1))
        raise ValueError(f"unsupported target type {target_type}")

    for _ in range(200):
        root = build()
        if root.depth() <= max_depth and root.complexity() <= max_complexity:
            return root
    raise ValueError("could not grow typed expression within constraints")


def random_program(
    rng: np.random.Generator,
    target_type: Type = Vec,
    max_depth: int = MAX_DEPTH,
    max_complexity: int = MAX_COMPLEXITY,
) -> Program:
    fallback = "pair_hist(pairs(mod(positions(w),10),1),10)" if target_type == Vec else "fourier_power(w,1)"
    for _ in range(300):
        root = _random_expr(rng, target_type, int(max_depth), float(max_complexity))
        try:
            return Program(_validate(root))
        except (TypeError, ValueError):
            continue
    return parse_program(fallback)


def mutate_program(rng: np.random.Generator, program: Program) -> Program:
    paths = _expr_paths(program.root)
    for _ in range(300):
        path = paths[int(rng.integers(0, len(paths)))]
        old = _expr_at(program.root, path)
        fixed_cost = program.root.complexity() - old.complexity()
        max_depth = MAX_DEPTH - len(path)
        max_complexity = MAX_COMPLEXITY - fixed_cost
        if max_depth < 1 or max_complexity < 0:
            continue
        try:
            new = _random_expr(rng, old.typ, max_depth, max_complexity)
            root = _validate(_replace_expr(program.root, path, new))
            if root.typ in (Vec, Scalar):
                return Program(root)
        except (TypeError, ValueError):
            continue
    return random_program(rng, program.root.typ)
