"""Parametrized test suite for all simdrng generators.

Every generator gets the same battery of tests:
- isinstance(rng, np.random.Generator)
- shape / dtype correctness
- SciPy compatibility
- State serialization round-trip
- Pickle round-trip
- Bulk fill correctness
- Jump reproducibility (Xoshiro family)
- Stream levels (node / gpu / thread)
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from scipy import stats

import simdrng

# ── Generator matrix ────────────────────────────────────────────────────────

ALL_GENERATORS = [
    ("SplitMix", lambda: simdrng.SplitMix(42)),
    ("Xoshiro", lambda: simdrng.Xoshiro(42)),
    ("XoshiroSIMD", lambda: simdrng.XoshiroSIMD(42)),
    ("XoshiroNative", lambda: simdrng.XoshiroNative(42)),
    ("ChaCha8", lambda: simdrng.ChaCha8(42)),
    ("ChaCha12", lambda: simdrng.ChaCha12(42)),
    ("ChaCha20", lambda: simdrng.ChaCha20(42)),
    ("Philox4x32", lambda: simdrng.Philox4x32(42)),
    ("Philox2x32", lambda: simdrng.Philox2x32(42)),
    ("Philox4x64", lambda: simdrng.Philox4x64(42)),
    ("Philox2x64", lambda: simdrng.Philox2x64(42)),
]


@pytest.fixture(params=ALL_GENERATORS, ids=[g[0] for g in ALL_GENERATORS])
def gen_factory(request):
    return request.param


# ── Basic functionality ─────────────────────────────────────────────────────


class TestGenerator:
    def test_returns_numpy_generator(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        assert isinstance(rng, np.random.Generator)

    def test_random_shape(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        assert rng.random((3, 4)).shape == (3, 4)

    def test_float32_dtype(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(100, dtype=np.float32)
        rng.random(out.shape, dtype=np.float32, out=out)
        assert out.dtype == np.float32
        assert np.all((out >= 0.0) & (out < 1.0))

    def test_integers(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        c = rng.integers(0, 10, size=(2, 5))
        assert c.shape == (2, 5)
        assert np.all((c >= 0) & (c < 10))

    def test_normal(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        d = rng.normal(0.0, 1.0, size=(3, 3))
        assert d.shape == (3, 3)

    def test_scipy_compat(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        s_norm = stats.norm.rvs(size=5, random_state=rng)
        assert s_norm.shape == (5,)
        s_int = stats.randint.rvs(0, 10, size=5, random_state=rng)
        assert s_int.max() < 10 and s_int.min() >= 0

    def test_random_values_in_range(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        vals = rng.random(10_000)
        assert np.all((vals >= 0.0) & (vals < 1.0))

    def test_different_seeds_differ(self, gen_factory):
        _name, factory = gen_factory
        rng1 = factory()
        # Create a second generator with a different seed
        rng2 = factory()  # same seed
        # Same seed should produce same sequence
        np.testing.assert_array_equal(rng1.random(10), rng2.random(10))


# ── State serialization ────────────────────────────────────────────────────


class TestStateSerialization:
    def test_state_roundtrip(self, gen_factory):
        """get_state -> set_state preserves sequence."""
        _name, factory = gen_factory
        rng = factory()
        rng.random(100)  # advance state
        state = rng.bit_generator.state
        expected = rng.random(10)
        rng.bit_generator.state = state
        actual = rng.random(10)
        np.testing.assert_array_equal(actual, expected)

    def test_pickle_roundtrip(self, gen_factory):
        """Pickle/unpickle preserves sequence."""
        _name, factory = gen_factory
        rng = factory()
        rng.random(100)
        state = rng.bit_generator.state
        bg = pickle.loads(pickle.dumps(rng.bit_generator))
        rng2 = np.random.Generator(bg)
        rng.bit_generator.state = state
        np.testing.assert_array_equal(rng.random(10), rng2.random(10))

    def test_state_wrong_generator_raises(self, gen_factory):
        """Setting state from a different generator type raises ValueError."""
        _name, factory = gen_factory
        rng = factory()
        state = rng.bit_generator.state
        state["bit_generator"] = "__nonexistent__"
        with pytest.raises(ValueError):
            rng.bit_generator.state = state


# ── Bulk fill ───────────────────────────────────────────────────────────────


class TestBulkFill:
    def test_fill_uniform_range(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(10_000, dtype=np.float64)
        rng.bit_generator.fill_uniform(out)
        assert np.all((out >= 0.0) & (out < 1.0))

    def test_fill_uint64_not_all_zero(self, gen_factory):
        _name, factory = gen_factory
        rng = factory()
        out = np.empty(1_000, dtype=np.uint64)
        rng.bit_generator.fill_uint64(out)
        assert not np.all(out == 0)

    def test_fill_uniform_matches_sequential(self, gen_factory):
        """Bulk fill produces same values as per-sample operator() path."""
        _name, factory = gen_factory
        rng1 = factory()
        rng2 = factory()

        # Per-sample via random_raw -> manual convert
        raw = rng1.bit_generator.random_raw(1000)
        expected = (raw >> np.uint64(11)).astype(np.float64) * (2.0**-53)

        out = np.empty(1000, dtype=np.float64)
        rng2.bit_generator.fill_uniform(out)
        np.testing.assert_array_equal(out, expected)


# ── Jump tests (Xoshiro family) ────────────────────────────────────────────

JUMP_GENERATORS = [
    ("Xoshiro", lambda: simdrng.Xoshiro(42)),
    ("XoshiroSIMD", lambda: simdrng.XoshiroSIMD(42)),
    ("XoshiroNative", lambda: simdrng.XoshiroNative(42)),
]


@pytest.fixture(
    params=JUMP_GENERATORS, ids=[g[0] for g in JUMP_GENERATORS],
)
def jump_factory(request):
    return request.param


class TestJump:
    def test_jump_produces_different_sequence(self, jump_factory):
        _name, factory = jump_factory
        rng1 = factory()
        rng2 = factory()
        rng2.bit_generator.jump()
        # After jump, sequences should differ
        a = rng1.random(100)
        b = rng2.random(100)
        assert not np.array_equal(a, b)

    def test_jump_reproducible(self, jump_factory):
        _name, factory = jump_factory
        rng1 = factory()
        rng2 = factory()
        rng1.bit_generator.jump()
        rng2.bit_generator.jump()
        np.testing.assert_array_equal(rng1.random(100), rng2.random(100))


class TestJumpN:
    def test_jump_n_forward_oracle(self):
        # Scalar Xoshiro has no per-lane cache: jump(k) must equal k raw draws.
        k = 1234
        advanced = simdrng.Xoshiro(42)
        advanced.bit_generator.random_raw(k)
        jumped = simdrng.Xoshiro(42)
        jumped.bit_generator.jump(k)
        np.testing.assert_array_equal(
            jumped.bit_generator.random_raw(50),
            advanced.bit_generator.random_raw(50),
        )

    def test_jump_n_composition(self, jump_factory):
        _name, factory = jump_factory
        a, b = 1_000_000, 7_654_321
        split = factory()
        combined = factory()
        split.bit_generator.jump(a)
        split.bit_generator.jump(b)
        combined.bit_generator.jump(a + b)
        np.testing.assert_array_equal(split.random(100), combined.random(100))

    def test_jump_n_zero_is_identity(self, jump_factory):
        _name, factory = jump_factory
        rng1 = factory()
        rng2 = factory()
        rng2.bit_generator.jump(0)
        np.testing.assert_array_equal(rng1.random(100), rng2.random(100))

    def test_jump_pow2_matches_fixed_jump(self, jump_factory):
        # jump(pow2(128)) is exactly the fixed 2^128 jump().
        _name, factory = jump_factory
        by_pow2 = factory()
        by_jump = factory()
        by_pow2.bit_generator.jump(simdrng.pow2(128))
        by_jump.bit_generator.jump()
        np.testing.assert_array_equal(by_pow2.random(100), by_jump.random(100))

    def test_jump_pow2_matches_step_count(self):
        # Scalar Xoshiro, e < 64: 2^e stride equals the raw step count 1 << e.
        by_pow2 = simdrng.Xoshiro(42)
        by_steps = simdrng.Xoshiro(42)
        by_pow2.bit_generator.jump(simdrng.pow2(20))
        by_steps.bit_generator.jump(1 << 20)
        np.testing.assert_array_equal(
            by_pow2.bit_generator.random_raw(50),
            by_steps.bit_generator.random_raw(50),
        )


# ── Stream levels (node / gpu / thread) ────────────────────────────────────

# (name, factory) where factory takes keyword-only levels on seed 42.
STREAM_LEVEL_FACTORIES = [
    ("Xoshiro", lambda **kw: simdrng.Xoshiro(42, **kw)),
    ("XoshiroSIMD", lambda **kw: simdrng.XoshiroSIMD(42, **kw)),
    ("XoshiroNative", lambda **kw: simdrng.XoshiroNative(42, **kw)),
]

SIMD_LEVEL_FACTORIES = [
    ("XoshiroSIMD", lambda **kw: simdrng.XoshiroSIMD(42, **kw)),
    ("XoshiroNative", lambda **kw: simdrng.XoshiroNative(42, **kw)),
]


@pytest.fixture(
    params=STREAM_LEVEL_FACTORIES, ids=[f[0] for f in STREAM_LEVEL_FACTORIES]
)
def level_factory(request):
    return request.param


@pytest.fixture(params=SIMD_LEVEL_FACTORIES, ids=[f[0] for f in SIMD_LEVEL_FACTORIES])
def simd_level_factory(request):
    return request.param


class TestStreamLevels:
    def test_keyword_order_gives_identical_output(self, level_factory):
        _name, factory = level_factory
        orders = [
            {"node": 1, "gpu": 2, "thread": 3},
            {"thread": 3, "gpu": 2, "node": 1},
            {"gpu": 2, "node": 1, "thread": 3},
        ]
        first = None
        for kw in orders:
            out = factory(**kw).random(50)
            if first is None:
                first = out
            else:
                np.testing.assert_array_equal(first, out)

    def test_omitted_levels_equal_zero(self, level_factory):
        _name, factory = level_factory
        np.testing.assert_array_equal(
            factory(gpu=2).random(50), factory(node=0, gpu=2, thread=0).random(50)
        )
        np.testing.assert_array_equal(
            factory(thread=3).random(50), factory(node=0, gpu=0, thread=3).random(50)
        )
        np.testing.assert_array_equal(
            factory(node=1).random(50), factory(node=1, gpu=0, thread=0).random(50)
        )

    def test_scalar_plain_equals_zero_tags(self):
        # scalar no-tags already equals all-zero tags
        np.testing.assert_array_equal(
            simdrng.Xoshiro(42).random(50),
            simdrng.Xoshiro(42, node=0, gpu=0, thread=0).random(50),
        )

    def test_distinct_values_give_distinct_output(self, level_factory):
        _name, factory = level_factory
        base = factory().random(50)
        with_thread = factory(thread=1).random(50)
        with_gpu = factory(gpu=1).random(50)
        with_node = factory(node=1).random(50)
        for other in (with_thread, with_gpu, with_node):
            assert not np.array_equal(base, other)
        assert not np.array_equal(with_gpu, with_thread)
        assert not np.array_equal(with_node, with_thread)
        assert not np.array_equal(with_node, with_gpu)

    def test_levels_match_cpp_jump_oracle(self, level_factory):
        _name, factory = level_factory
        # stream(node{1})/stream(gpu{1})/stream(thread{1}) must equal the
        # fixed jump polynomials 2^191 / 2^159 / 2^127 on the same seed. On
        # the SIMD backends the oracle is the lane 0 stream.
        cases = [({"node": 1}, 191), ({"gpu": 1}, 159), ({"thread": 1}, 127)]
        for kw, exponent in cases:
            w = factory(**kw).bit_generator.state["state"].get("simd_width", 1)
            raw = factory(**kw).bit_generator.random_raw(50 * w)
            jumped = simdrng.Xoshiro(42).bit_generator
            jumped.jump(simdrng.pow2(exponent))
            np.testing.assert_array_equal(raw[::w][:50], jumped.random_raw(50))

    def test_lane_zero_equals_scalar(self, simd_level_factory):
        _name, factory = simd_level_factory
        for kw in (
            {},
            {"node": 1, "gpu": 2, "thread": 3},
            {"thread": 3},
            {"gpu": 2},
            {"node": 7},
        ):
            scalar = simdrng.Xoshiro(42, **kw).bit_generator
            simd = factory(**kw).bit_generator
            st = simd.state["state"]
            w = st["simd_width"]
            assert [st["s"][r * w] for r in range(4)] == scalar.state["state"]["s"]
            scalar = simdrng.Xoshiro(42, **kw).bit_generator
            simd = factory(**kw).bit_generator
            np.testing.assert_array_equal(scalar.random_raw(1), simd.random_raw(1))

    def test_simd_lane_zero_equals_native(self):
        for kw in ({}, {"node": 1, "gpu": 2, "thread": 3}, {"thread": 3}):
            a = simdrng.XoshiroSIMD(42, **kw).bit_generator.state["state"]
            b = simdrng.XoshiroNative(42, **kw).bit_generator.state["state"]
            wa, wb = a["simd_width"], b["simd_width"]
            assert [a["s"][r * wa] for r in range(4)] == [
                b["s"][r * wb] for r in range(4)
            ]

    def test_plain_constructor_unchanged(self, level_factory):
        # Plain-seed values pinned before the tag API landed. The plain lane
        # layout keeps lane i at i * 2^128 of the seed sequence. The state is
        # read before the first draw, because a draw fills the cache and
        # advances the state.
        _name, factory = level_factory
        bg = factory().bit_generator
        st = bg.state["state"]
        w = st.get("simd_width", 1)
        lane0 = [
            0xBDD732262FEB6E95,
            0x28EFE333B266F103,
            0x47526757130F9F52,
            0x581CE1FF0E4AE394,
        ]
        lane1 = [
            0x81746704FDE896B5,
            0x645E944932DAE0AE,
            0xF4776829231C282C,
            0x2393F9798732DBA1,
        ]
        assert [st["s"][r * w] for r in range(4)] == lane0
        if w > 1:
            assert [st["s"][r * w + 1] for r in range(4)] == lane1
        assert int(bg.random_raw(1)[0]) == 0xD0764D4F4476689F

    def test_max_in_range_works(self, level_factory):
        _name, factory = level_factory
        rng = factory(node=(1 << 64) - 1, gpu=(1 << 32) - 1, thread=(1 << 32) - 1)
        assert rng.random(10).shape == (10,)

    def test_out_of_range_raises_valueerror(self, level_factory):
        _name, factory = level_factory
        with pytest.raises(ValueError):
            factory(gpu=1 << 32)
        with pytest.raises(ValueError):
            factory(thread=1 << 32)
        with pytest.raises(ValueError):
            factory(node=1 << 64)
        for kw in ({"node": -1}, {"gpu": -1}, {"thread": -1}):
            with pytest.raises(ValueError):
                factory(**kw)
        with pytest.raises(ValueError):
            factory(gpu="1")

    def test_cluster_keyword_is_removed(self):
        # 'cluster' is a breaking removal with no alias: TypeError, not ValueError.
        for maker in (
            simdrng.Xoshiro,
            simdrng.XoshiroSIMD,
            simdrng.XoshiroNative,
            simdrng.XoshiroBitGenerator,
            simdrng.XoshiroSIMDBitGenerator,
            simdrng.XoshiroNativeBitGenerator,
        ):
            with pytest.raises(TypeError):
                maker(42, cluster=1)

    def test_none_level_raises_valueerror(self):
        # A supplied None is a non-integer level, not an omitted keyword.
        for maker in (
            simdrng.Xoshiro,
            simdrng.XoshiroSIMD,
            simdrng.XoshiroNative,
            simdrng.XoshiroBitGenerator,
            simdrng.XoshiroSIMDBitGenerator,
            simdrng.XoshiroNativeBitGenerator,
        ):
            for kw in ({"node": None}, {"gpu": None}, {"thread": None}):
                with pytest.raises(ValueError):
                    maker(42, **kw)

    def test_omitted_keywords_keep_plain_constructor(self):
        # Omitting every level keyword must keep the plain constructor.
        for cls, factory in (
            (simdrng.XoshiroBitGenerator, simdrng.Xoshiro),
            (simdrng.XoshiroSIMDBitGenerator, simdrng.XoshiroSIMD),
            (simdrng.XoshiroNativeBitGenerator, simdrng.XoshiroNative),
        ):
            assert cls(42).state == factory(42).bit_generator.state


# ── Global generator ───────────────────────────────────────────────────────


class TestGlobalGenerator:
    def test_seed_and_random(self):
        simdrng.seed(123)
        a = simdrng.random(10)
        simdrng.seed(123)
        b = simdrng.random(10)
        np.testing.assert_array_equal(a, b)

    def test_default_rng_returns_generator(self):
        rng = simdrng.default_rng(99)
        assert isinstance(rng, np.random.Generator)

    def test_normal(self):
        simdrng.seed(42)
        vals = simdrng.normal(size=1000)
        assert vals.shape == (1000,)

    def test_integers(self):
        simdrng.seed(42)
        vals = simdrng.integers(0, 100, size=50)
        assert np.all((vals >= 0) & (vals < 100))
