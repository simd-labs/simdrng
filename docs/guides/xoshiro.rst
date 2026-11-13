Xoshiro256++
============

What it is
----------

xoshiro256++ is Blackman & Vigna's all-purpose 64-bit generator: a small,
extremely fast scrambled-linear engine with a period of
2\ :sup:`256` − 1 that passes the full BigCrush / PractRand batteries. It is the
default ``simdrng::Xoshiro``.

When to use it
--------------

Reach for xoshiro256++ as the **general-purpose default** — simulation,
sampling, games, procedural generation — anywhere you want maximum throughput
of high-quality 64-bit values and do not need cryptographic guarantees or
stateless counter-based seeking (use :doc:`philox` for the latter).

Scalar, SIMD dispatch, and native
----------------------------------

Three implementations share one interface; pick by include and type:

.. list-table::
   :header-rows: 1
   :widths: 26 30 44

   * - Type
     - Header
     - Use when
   * - ``XoshiroScalar``
     - ``<simdrng/xoshiro_scalar.hpp>``
     - Portable, dependency-free, ``constexpr``-friendly single-stream use.
   * - ``XoshiroSIMD``
     - ``<simdrng/xoshiro_simd.hpp>``
     - One binary, many CPUs: the SIMD tier is chosen at runtime via xsimd
       dispatch (AVX-512 / AVX2 / SSE2 / NEON / …).
   * - ``XoshiroNative``
     - ``<simdrng/xoshiro_simd.hpp>``
     - You build with ``-march=native`` and want the widest width inlined with
       no dispatch indirection.

``simdrng::Xoshiro`` (from ``<simdrng/xoshiro.hpp>``) aliases ``XoshiroSIMD``
when built with xsimd, and ``XoshiroScalar`` otherwise.

.. code-block:: cpp

   #include <simdrng/xoshiro.hpp>

   simdrng::Xoshiro rng(42);          // seed (SplitMix64-expanded internally)
   std::uint64_t x = rng();           // next 64-bit value
   double u = rng.uniform();          // double in [0, 1)

The SIMD types refill an internal cache with wide batches, so an ordinary
generate loop runs vectorised — there is no separate bulk-fill API.

Periods and streams
-------------------

xoshiro256++ has a period of 2\ :sup:`256` − 1. The seed is expanded through
SplitMix64 (see :doc:`splitmix`) so even low-entropy seeds give sound state. For
disjoint parallel streams, use the level tags ``simdrng::node``,
``simdrng::gpu`` and ``simdrng::thread``:

.. code-block:: cpp

   simdrng::XoshiroScalar rng{seed, simdrng::node{n}, simdrng::gpu{g}, simdrng::thread{t}};

The levels are jump-ahead positions in the one sequence the seed gives.
``node{n}`` starts the stream at ``n * 2^191`` outputs, ``gpu{g}`` at
``g * 2^159``, and ``thread{t}`` at ``t * 2^127``. Each
stream owns 2\ :sup:`127` outputs, so one seed gives disjoint streams. A
Xoshiro256++ stream that draws more than 2\ :sup:`127` outputs runs into the
next stream's numbers, and nothing checks for this. The
strides align with the fixed jumps: ``thread{2}`` equals ``jump()``,
``gpu{2}`` equals ``mid_jump()``, and ``node{2}`` equals ``long_jump()``.
``node`` has no value limit. ``gpu`` and ``thread`` hold 32 bits each. An
omitted level is ``0``. A level value above its max stops the program. See
:ref:`gen-properties` for the jump spacing and the full period table.

``XoshiroSIMD`` and ``XoshiroNative`` accept the same tags. A tag tuple
selects the same block of 2\ :sup:`127` outputs as ``XoshiroScalar``. The
lanes split that block. Lane 0 starts at the block start and equals
``XoshiroScalar`` with the same seed and tags. Lane i starts at ``i * S``
into the block, with ``S = 2^(127 - ceil(log2(W)))`` for W lanes. Each lane
owns S outputs. W is 2, 4 or 8 on the supported targets, so S is
2\ :sup:`126`, 2\ :sup:`125` or 2\ :sup:`124`. A plain-seeded
``XoshiroSIMD`` or ``XoshiroNative`` does not follow the tag layout. It
keeps lane i at ``i * 2^128``, and those lanes overlap scalar ``thread``
streams of the same seed. A program that mixes the two layouts applies tags
to each generator.

To address an arbitrary stream offset, ``jump(n)`` advances the state by exactly
``n`` calls to the generator. It builds the jump polynomial ``x^n mod P(x)`` on
the fly (square-and-multiply in GF(2)[x]), so any ``n`` in ``[0, 2^64)`` works.
For power-of-two strides (including those beyond ``2^64``), pass a ``pow2`` tag:
``jump(pow2{e})`` advances by ``2^e`` — the same convention as the fixed jumps,
so ``jump(pow2{128})`` is identical to ``jump()``.

.. code-block:: cpp

   rng.jump(offset);        // skip ahead `offset` outputs in O(log offset)
   rng.jump(pow2{200});     // skip ahead 2^200 outputs
   rng.jump(pow2{128});     // == rng.jump()

References
----------

xoshiro256++ and its SplitMix64 seeding follow D. Blackman and S. Vigna,
*Scrambled Linear Pseudorandom Number Generators* (the design and the ``++``
scrambler) and S. Vigna's https://prng.di.unimi.it/ (seeding and the shootout).
See :doc:`/references`.
