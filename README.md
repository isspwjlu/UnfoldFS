# UnfoldFS — Supercell band unfolding → Fermi surfaces in the primitive BZ

**UnfoldFS** unfolds the result of a **supercell VASP calculation** back into the
**primitive first Brillouin zone** and turns it into a Fermi surface carrying the
spectral weight `A(k, E_F)`, written in formats that **VESTA / XCrySDen /
FermiSurfer** open directly.

* Algorithm, conventions and every measured number → [`docs/algorithms.md`](docs/algorithms.md)
* Field-by-field specification of the output files → [`docs/formats.md`](docs/formats.md)

---

## 0. Repository and paper

* Code: **https://github.com/isspwjlu/UnfoldFS**
* Paper: *UnfoldFS: unfolded Fermi surfaces from supercell band calculations*
  (arXiv: to be announced)
* License: MIT (`LICENSE`)

The package carries **two projection bases**, and the two papers in this
repository exercise them separately:

| route | projection | weights | where |
|---|---|---|---|
| **plane wave** | the plane-wave coefficients of `WAVECAR` | the strict Popescu–Zunger weights; exact, no atomic correspondence needed | `paper_pw/` |
| **partial wave** | the partial-wave coefficients of `PROCAR` | orbital-resolved, but the basis is neither orthogonal nor complete | `paper/` |

The plane-wave route is the one the tool was developed for: its weights
satisfy a **completeness identity** (Σ_j w_j = 1 for every state), which
makes the implementation testable without any reference calculation, and
the reconstructed spectrum reproduces the spectral weight of a direct
primitive-cell calculation to 0.2 %. See §2.2 and
[`docs/algorithms.md`](docs/algorithms.md) §17.

The public version ships without the test data, the computed results and the
manuscript; the tests that depend on them are skipped automatically. Reproducing
the paper's examples means running VASP yourself (scripts in `scripts/cdw/` and
`scripts/figures/`).

## 1. Installation

```bash
pip install -e .
```

That gives you the `unfoldfs` command (the package is called `unfoldfs`; the
project is UnfoldFS). If you prefer not to install the package,
the module entry point is equivalent (`python -m unfoldfs.cli --help`).

| Dependency | Requirement | Notes |
|---|---|---|
| Python | ≥ 3.9 | |
| numpy / scipy | ≥ 1.20 / ≥ 1.6 | periodic KD-tree for grid-point matching |
| spglib | **≥ 2.0** | primitive-cell point group (the code uses the 2.x dict API) |
| matplotlib | **optional** | only needed for the PNG of `unfoldfs band` (`pip install 'unfoldfs[plot]'`) |
| VASP | 6.6.0 (tested) | only used to produce the inputs (`EIGENVAL` / `PROCAR` / `DOSCAR` / `WAVECAR`) |

## 2. The five subcommands

```bash
unfoldfs gen    --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in
                [--resolution R] [--grid P1 P2 P3] [--grid-type {gamma,mp}]
                [--outdir DIR] [--nbands N] [--encut E] [--kpar K]
unfoldfs unfold --dir DIR [--ef {auto,vbm,doscar,value}] [--delta D] [--sigma S]
                [--wavecar WAVECAR] [--min-dominance X]
                [--strict-dominance] [--no-symmetrize] [--orbitals] [--outdir DIR]
unfoldfs fs     --npz A(k,E).npz --prim PRIMCELL.vasp
                [--mode {crest,spectral}] [--symprec S] [--verify OTHER.npz]
                [--velocity] [--rays N] [--out OUT.npz]
unfoldfs band   --supercell … --primcell … --transmat … --kpath KPATH.in --outdir DIR
                # stage 2: unfoldfs band --dir DIR --outdir DIR2 [--e-min E] [--e-max E]
                #          [--wavecar WAVECAR] [--mirror] [--no-plot] [--no-symmetrize]
unfoldfs info   --dir DIR [--grid-map GRID_MAP]
```

* **`gen`** — builds the folded grid and a `KPOINTS` file (already deduplicated;
  **VASP 6.6.0 crashes on an explicit list that repeats a k-point**), plus
  `grid_map.npz` (needed by `unfold`) and `INCAR.suggested` (with `ISYM = 0` and
  `LORBIT = 12`).
* **`unfold`** — unfolds and writes the Fermi-surface files. `--ef auto` first
  decides whether the system is metallic: for a metal it takes the E_F from
  DOSCAR, otherwise `VBM + delta`. `--wavecar WAVECAR` selects the
  **plane-wave** route (§2.2); without it the partial-wave route is used.
  `--min-dominance X` lowers the single-G dominance gate, which is needed
  for states that are not eigenstates of the primitive translations (a
  charge-density wave, a vacancy). `--orbitals` additionally computes the
  **orbital-resolved** spectral weights (written to the npz as `band_orb`;
  ⚠ the orbital weights do **not** sum to the total — there are cross terms —
  so normalise by `band_orb / Σ_orb band_orb`; partial-wave route only).
* **`fs`** — the *last step*: extract the Fermi surface from the spectral
  function, diagnose it, and check it. See §2.1.
* **`band`** — unfolded bands along a k path, in two stages: read `KPATH.in` and
  write the VASP inputs, then read the VASP output and unfold.
* **`info`** — prints the grid specification, the number of irreducible points,
  which symmetry operations were kept or dropped and why, `det(M)`, and more.

### 2.1 `unfoldfs fs` — the Fermi surface, its diagnostics, and the check

```bash
unfoldfs fs --npz out/A(k,E).npz --prim PRIMCELL.vasp
```

This is the step that turns a spectral function into a Fermi surface, and the
step where most of the silent errors live:

* **the crest, not the isosurface** — `A(k,E)` at E_F is a Gaussian *ridge*;
  its crest (the band that crosses E_F) is the Fermi surface, while the
  isosurface `A(k,E_F) = const` gives the ridge's two **flanks**. `--mode crest`
  (default) tracks the crest through the dominant peak with three gates — a
  genuine local maximum, |E*(k) − E_F| < 0.25 eV, and `A(k,E_F) > 0.02·max A`
  — each of which prevents a specific artefact (a phantom loop where
  `A(k,E_F) = 0`). `--mode spectral` reproduces the customary isosurface, for
  comparison only.
* **covariance diagnostic** — reports the relative dispersion δ(R) of
  `A(k,E_F)` under the twelve primitive point-group operations. δ ≫ 0 means
  the unfolded weights are not symmetric (see §6 and the paper). The command
  also prints, when `--verify` is given, a **floor** for δ(R): the value
  measured on a *direct* primitive-cell calculation, which is finite because
  of the DFT numerics alone. Comparing the two separates "the structure
  breaks the symmetry" (δ ≈ floor) from "the projection introduces the
  artefact" (δ ≫ floor) — only the excess is the unfolding's error.
  On the plane-wave route the excess is at the level of the floor
  (2 × 10⁻⁵ against a floor of 6.7 × 10⁻⁵ for the 3×1×1 supercell of
  1T-TaS₂).
* **verification protocol** — `--verify DIRECT.npz` compares the unfolded Fermi
  surface with a direct primitive-cell calculation: the fraction of the curve
  that coincides, plus the median/max point-to-segment distance in units of the
  mesh spacing. Unfolding redistributes weight but must not *move* the Fermi
  surface.
* **Fermi velocity** — `--velocity` follows the crest along rays from Γ and
  gives |v_F| at E_F (using the crest field, not the per-band index, which
  jumps between k points in an unfolded supercell).

Run any command with `--help` for the full parameter list.

### 2.2 The two projection bases

The unfolding assigns a supercell state to a primitive momentum by
projecting it, and the projection can be carried out in two bases.

**Plane wave** (`--wavecar WAVECAR`, the default recommendation). The
projection is made on the plane waves of the supercell, which form an
orthonormal and complete basis, and the weight of a candidate momentum
is the square of the state's projection onto the primitive Bloch states:
the strict construction of Popescu and Zunger (PRB **85**, 085201 (2012)).
Three things follow, and they are what the plane-wave route was built
for:

* the weights of **all** candidate momenta are available at once, so they
  satisfy the **completeness identity** Σ_j w_j = 1 for every state —
  an implementation test that needs no reference calculation (measured
  deviation < 2 × 10⁻⁷);
* **no atomic correspondence is needed**, which matters when the supercell
  contains a vacancy or an alloy;
* the candidates form a **partition** of the plane-wave basis, so no
  candidate can be missed or counted twice. (The common alternative — take
  the plane waves that belong to the primitive reciprocal lattice and
  translate the whole set by one vector — leaves part of the translated set
  outside the cutoff sphere; on the 3×1×1 supercell of 1T-TaS₂ about
  **26 %** of the candidates are affected that way.)

**Partial wave** (no `--wavecar`). The projection is built from the
partial-wave coefficients of `PROCAR` and is resolved per orbital, so
the weights are orbital-resolved. The basis is neither orthogonal nor
complete, so the weights are not exact spectral weights; they are
comparable within one calculation only. See §6.

### One default behaviour of `unfold`

The unfolded spectral weight `A(k,E)` is **not covariant** under the crystal
point group — two symmetry-related k points can differ by 100 %. Two causes
contribute: an operation that does not preserve the supercell lattice folds
symmetry-related momenta onto different sets of supercell momenta (this
affects every unfolding method), and the replica average, which is taken over
a single representative atom and is therefore not equivariant under a
permutation of the atoms of the supercell. The isosurface of a Fermi surface built from it is therefore
distorted with respect to the crystal symmetry. **By default `A(k,E)` is
orbit-averaged over the full primitive-cell point group**, which brings the
measured relative deviation from 1.000e+00 down to 0.000e+00. `--no-symmetrize`
turns this off. The per-band data
(`band.N`, `FS.bxsf`, `band_e`/`band_w` in the npz) is **not** symmetrised:
symmetry-related points simply do not have the same set of bands.

## 3. Typical workflow

```bash
# ① build the grid and KPOINTS
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp \
              --transmat TRANSMAT.in --grid 6 6 1 --outdir run/

# ② run VASP (ISYM = 0 and LORBIT = 12 from INCAR.suggested must be kept;
#              the plane-wave route also needs LWAVE = .TRUE.)
cp run/INCAR.suggested run/INCAR
cp POTCAR run/POTCAR
cd run && mpirun -np 8 vasp_std

# ③ unfold (add --wavecar run/WAVECAR for the plane-wave route)
unfoldfs unfold --dir run/ --outdir fs/ --ef vbm

# ④ open the Fermi surface
```

The plane-wave route needs the plane-wave coefficients, so the INCAR must
carry `LWAVE = .TRUE.` and the file `WAVECAR` must be kept:

```bash
unfoldfs unfold --dir run/ --wavecar run/WAVECAR --outdir fs/ --ef vbm
```

`WAVECAR` is large — 11 GB for the 3×1×1 supercell of 1T-TaS₂ on a
96 × 96 mesh (3072 k points) — and VASP does not write it by default. A
calculation made without it cannot be unfolded by this route afterwards.

| Tool | File to open | What you get |
|---|---|---|
| **XCrySDen** | `FS.bxsf` | Fermi surface from the per-band energy grid (**Γ grid only**) |
| **FermiSurfer** | `band.N` | per-band Fermi surfaces (energies relative to E_F, open as they are) |
| **VESTA** | `FS.cube` | the `A(k, E_F)` **volumetric field** — ⚠ **not** the Fermi surface: its isosurface is the ridge's two flanks (see §6.4) |
| Python | `A(k,E).npz` | the full spectral function plus metadata, for your own analysis |

⚠ **For a supercell unfolding, prefer `unfoldfs fs`** (see §2.1): the per-band
files above are only a clean Fermi surface for the **primitive cell** — in an
unfolded supercell the band index jumps between k points (the same band
number is a different replica at different k), so per-band contours break into
fragments. `unfoldfs fs` tracks the crest of the ridge instead, with the three
gates that keep phantom loops out.

Unfolded bands follow a separate pipeline:

```bash
unfoldfs band --supercell POSCAR --primcell PRIMCELL.vasp \
              --transmat TRANSMAT.in --kpath KPATH.in --outdir run/
# ... run VASP (same as above) ...
unfoldfs band --dir run/ --outdir bandfs/
```

The files `ENERGY.grd` / `MOMENTUM.grd` / `WEIGHT.grd` / `KLABELS` in `bandfs/`
have **exactly the names and layout of the vaspkit ones**, so an existing MATLAB
plotting script works on them unchanged.

## 4. Output files

| File | Consumer | Content |
|---|---|---|
| `FS.cube` | VESTA / XCrySDen / VMD | Gaussian cube: the volume data is `A(k, E_F)`; the header gives E_F and the suggested isosurface level |
| `FS.bxsf` | XCrySDen / FermiSurfer | per-band energy grid (**Γ grid only**) |
| `band.N` | FermiSurfer | FermiSurfer `.frmsf` per-band files (one band each, energies relative to E_F) |
| `A(k,E).npz` | yourself | spectral function volume data, grids, per-band energies/weights, metadata |
| `ENERGY/MOMENTUM/WEIGHT.grd`, `KLABELS` | vaspkit / MATLAB | the four unfolded-band files, same names and layout as vaspkit's |

> The numbering of `band.N` is by **ascending energy**: `band.1` is the lowest
> deep band, **not** the first band crossing E_F. The `band_e`/`band_w` arrays
> in the npz contain `+inf` padding slots — always mask them with `valid` when
> doing your own `argmin` or minimum.

## 5. Pitfalls that silently corrupt a Fermi surface

| Symptom | What to do |
|---|---|
| `error: VASP computed N k-points != grid_map records M` | `ISYM ≠ 0` in the INCAR makes VASP reduce the explicit k list to the IBZ ⇒ keep **`ISYM = 0`** |
| No phase block in PROCAR | `LORBIT = 11` does not write phases ⇒ keep **`LORBIT = 12`** |
| `warning: NBANDS may be too small` | check `NELECT` in OUTCAR and override with `gen --nbands N` |
| `error: Error reading KPOINTS file` | `ISMEAR ≤ -4` (tetrahedron method) expects VASP to build the grid itself ⇒ use `ISMEAR = 1` + `SIGMA = 0.2` |
| `error: single-G dominance 0.49 < 0.5` | the grid is too coarse, the supercell BZ is undersampled ⇒ refine the grid |
| `error: the selected E_F lies inside the gap …` | wrong E_F for a semiconductor ⇒ use `--ef vbm` or `--ef <value>` |
| ⚠ **a wrong phase convention deforms the whole Fermi surface** (not merely "not reproducible") | the convention is fixed by `s1` (physically unique) plus `s2` (picked by single-G concentration); when the degeneracy warning fires, **do not trust the Fermi-surface shape** of that run ⇒ refine the grid |
| `error: WAVECAR 不存在` | the plane-wave route needs `LWAVE = .TRUE.` in the INCAR and the file kept next to the calculation |
| the plane-wave weights do not sum to 1 | the candidate set is built by equivalence class, so this cannot happen; if the identity fails, the reconstruction of the plane-wave basis is wrong (see `docs/algorithms.md` §17) |

## 6. Known limitations (the short version)

1. **No SOC** (`ISPIN = 1`) — gap values are a baseline only.
2. The k sampling of a metallic path is still sparse, so a **quantitative
   Fermi-surface shape** needs a denser sampling of the supercell BZ; the
   energies, the connectivity of the bands, the topology (how many pockets) and
   the choice of E_F are reliable.
3. Per-band data is not symmetrised ⇒ per-band weights are not quantitative.
4. **The isosurface of `A(k,E_F)` is NOT the Fermi surface** — it gives the two
   **flanks** of the Gaussian ridge, not its crest. `FS.cube` *is* `A(k,E_F)`
   (as a volume), so isosurfacing it gives the flanks too. Draw the Fermi
   surface from `FS.bxsf` / `band.N` (per-band energies ⇒ E = E_F contour),
   or better use `unfoldfs fs`, which extracts the crest directly (§2.1).
5. The weights in `WEIGHT.grd` are **Bloch-character G-resolved weights**, not the
   WAVECAR plane-wave expansion |c(k)|² of vaspkit 282; the physics differs and
   the numbers are not comparable.
6. **The plane-wave route requires `WAVECAR`**, which VASP writes only when
   `LWAVE = .TRUE.` and which is large (11 GB for the 3×1×1 supercell of
   1T-TaS₂ on a 96 × 96 mesh). A calculation made without it cannot be
   unfolded by this route afterwards. The route also does not deliver
   orbital-resolved weights: obtaining them from plane waves requires a
   stricter, atom-centred projection (Quan et al., PRB **113** (2026)).
7. **The partial-wave route** (the default, without `--wavecar`) evaluates the
   projection from the **partial-wave coefficients of the PROCAR file**,
   which are neither orthogonal nor complete. The weights are therefore
   **not exact spectral weights** and are only comparable within a single
   calculation.
   The covariance defect of §2.1 has two parts, separated by a per-operation
   test: operations that do not preserve the supercell lattice break the
   folding itself (this happens in any unfolding method), while operations
   that *do* preserve it still give δ(R) ≈ 0.74, because the replica average
   is taken over a single representative atom and is not equivariant under a
   permutation of the atoms of the supercell. The second part is specific to
   this implementation, and it is isolated completely by the √3×√3 supercell
   of 1T-TaS₂, whose space group is that of the primitive cell: all twelve
   primitive operations leave its lattice invariant, so the folding is
   equivariant for each of them, and the dispersion still reaches δ(R) = 0.99.
   On the plane-wave route this second part is absent — the weights are the
   strict ones — and the residual asymmetry is confined to the 0.26 % of the
   mesh points where a state is shared between two candidate momenta.

The complete list and all measured numbers → [`docs/algorithms.md`](docs/algorithms.md) §14.

## 7. Repository layout

```
unfoldfs/            the tool itself (the installable UnfoldFS package)
  geometry.py        structure parsing / transform-convention detection / candidate G / common symmetry set
  grid.py            uniform grid / symmetric orbit decomposition / k-point dedup / KPOINTS writer
  vaspio.py          EIGENVAL / DOSCAR / PROCAR (LORBIT=12) parsing
  wfc.py             WAVECAR reader + reconstruction of the plane-wave basis (FFT grid / ordering)
  pwunfold.py        the plane-wave (Popescu–Zunger) unfolding: equivalence classes, weights, completeness identity
  unfold.py          Bloch-character projection unfolding / replica phase restoration and averaging / candidate recombination
  expand.py          symmetric expansion onto a uniform grid / spectral function A(k,E)
  fs.py              the Fermi surface: crest field, the three gates, covariance diagnostic, verification, velocity
  fsout.py           FS.cube / FS.bxsf / band.N / A(k,E).npz writer
  band.py            KPATH.in parsing / unfolded-band assembly / the four vaspkit files / plotting
  cli.py             gen / unfold / band / fs / info
docs/                algorithms.md (algorithm and measured numbers) / formats.md (file formats)
tests/                unit and integration tests
scripts/figures/      figure scripts (see its own README for the conventions)
scripts/analysis/     data-analysis scripts, no figures (see its own README)
scripts/cdw/          the CDW example (build the supercell / check the structure / compare unfolded results)
                     build_trimer_cdw.py = C3-covariant trimerisation (radial shrink); build_cdw_tas2.py has a non-C3-covariant F3m field, see its docstring
examples/             archive: exploration-stage prototypes and server job scripts (not part of the package)
```

The scripts under `scripts/` and `examples/` are not part of the UnfoldFS
package (they are not installed).
