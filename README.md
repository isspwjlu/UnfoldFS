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
* Paper: *Unfolded Fermi surfaces from supercell band calculations: extracting,
  validating, and rendering the Fermi surface of a supercell spectral function*
  (arXiv: to be announced)
* License: MIT (`LICENSE`)

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
| VASP | 6.6.0 (tested) | only used to produce the inputs (`EIGENVAL` / `PROCAR` / `DOSCAR`) |

## 2. The four subcommands

```bash
unfoldfs gen    --supercell POSCAR --primcell PRIMCELL.vasp --transmat TRANSMAT.in
                [--resolution R] [--grid P1 P2 P3] [--grid-type {gamma,mp}]
                [--outdir DIR] [--nbands N] [--encut E] [--kpar K]
unfoldfs unfold --dir DIR [--ef {auto,vbm,doscar,value}] [--delta D] [--sigma S]
                [--strict-dominance] [--no-symmetrize] [--outdir DIR]
unfoldfs band   --supercell … --primcell … --transmat … --kpath KPATH.in --outdir DIR
                # stage 2: unfoldfs band --dir DIR --outdir DIR2 [--e-min E] [--e-max E]
                #          [--mirror] [--no-plot] [--no-symmetrize]
unfoldfs info   --dir DIR [--grid-map GRID_MAP]
```

* **`gen`** — builds the folded grid and a `KPOINTS` file (already deduplicated;
  **VASP 6.6.0 crashes on an explicit list that repeats a k-point**), plus
  `grid_map.npz` (needed by `unfold`) and `INCAR.suggested` (with `ISYM = 0` and
  `LORBIT = 12`).
* **`unfold`** — unfolds and writes the Fermi-surface files. `--ef auto` first
  decides whether the system is metallic: for a metal it takes the E_F from
  DOSCAR, otherwise `VBM + delta`.
* **`band`** — unfolded bands along a k path, in two stages: read `KPATH.in` and
  write the VASP inputs, then read the VASP output and unfold.
* **`info`** — prints the grid specification, the number of irreducible points,
  which symmetry operations were kept or dropped and why, `det(M)`, and more.

Run any command with `--help` for the full parameter list.

### One default behaviour of `unfold`

The unfolded spectral weight `A(k,E)` is **not covariant** under the crystal
point group — two symmetry-related k points can differ by 100 %. The
isosurface of a Fermi surface built from it is therefore distorted with respect
to the crystal symmetry. **By default `A(k,E)` is orbit-averaged over the full
primitive-cell point group**, which brings the measured relative deviation from
1.000e+00 down to 0.000e+00. `--no-symmetrize` turns this off. The per-band data
(`band.N`, `FS.bxsf`, `band_e`/`band_w` in the npz) is **not** symmetrised:
symmetry-related points simply do not have the same set of bands.

## 3. Typical workflow

```bash
# ① build the grid and KPOINTS
unfoldfs gen --supercell POSCAR --primcell PRIMCELL.vasp \
              --transmat TRANSMAT.in --grid 6 6 1 --outdir run/

# ② run VASP (ISYM = 0 and LORBIT = 12 from INCAR.suggested must be kept)
cp run/INCAR.suggested run/INCAR
cp POTCAR run/POTCAR
cd run && mpirun -np 8 vasp_std

# ③ unfold
unfoldfs unfold --dir run/ --outdir fs/ --ef vbm

# ④ open the Fermi surface
```

| Tool | File to open | What you get |
|---|---|---|
| **VESTA** | `FS.cube` | the `A(k, E_F)` volumetric isosurface (the suggested level is written in the second line of the header) |
| **XCrySDen** | `FS.bxsf` | Fermi surface from the per-band energy grid (**Γ grid only**) |
| **FermiSurfer** | `band.N` | per-band Fermi surfaces (energies relative to E_F, open as they are) |
| Python | `A(k,E).npz` | the full spectral function plus metadata, for your own analysis |

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

## 6. Known limitations (the short version)

1. **No SOC** (`ISPIN = 1`) — gap values are a baseline only.
2. The k sampling of a metallic path is still sparse, so a **quantitative
   Fermi-surface shape** needs a denser sampling of the supercell BZ; the
   energies, the connectivity of the bands, the topology (how many pockets) and
   the choice of E_F are reliable.
3. Per-band data is not symmetrised ⇒ per-band weights are not quantitative.
4. **Draw the Fermi surface from `FS.bxsf` / `FS.cube`**, not from a heat map of
   `A(k,E_F)` — the latter gives the two flanks of the Gaussian ridge rather
   than its crest.
5. The weights in `WEIGHT.grd` are **Popescu G-resolved weights**, not the
   WAVECAR plane-wave expansion |c(k)|² of vaspkit 282; the physics differs and
   the numbers are not comparable.

The complete list and all measured numbers → [`docs/algorithms.md`](docs/algorithms.md) §14.

## 7. Repository layout

```
unfoldfs/            the tool itself (the installable UnfoldFS package)
  geometry.py        structure parsing / transform-convention detection / candidate G / common symmetry set
  grid.py            uniform grid / symmetric orbit decomposition / k-point dedup / KPOINTS writer
  vaspio.py          EIGENVAL / DOSCAR / PROCAR (LORBIT=12) parsing
  unfold.py          Popescu-projection unfolding / replica phase restoration and averaging / candidate recombination
  expand.py          symmetric expansion onto a uniform grid / spectral function A(k,E)
  fsout.py           FS.cube / FS.bxsf / band.N / A(k,E).npz writer
  band.py            KPATH.in parsing / unfolded-band assembly / the four vaspkit files / plotting
  cli.py             gen / unfold / band / info
docs/                algorithms.md (algorithm and measured numbers) / formats.md (file formats)
tests/                unit and integration tests
scripts/figures/      figure scripts (see its own README for the conventions)
scripts/cdw/          the CDW example (build the supercell / check the structure / compare unfolded results)
examples/             archive: exploration-stage prototypes and server job scripts (not part of the package)
```

The scripts under `scripts/` and `examples/` are not part of the UnfoldFS
package (they are not installed).
