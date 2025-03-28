# Changelog

These release notes are automatically generated from commits and GitHub issues and PRs.
If it is out of date, please run

GITHUB_TOKEN=$(gh auth token) uv run .github/generate-release-notes.py

## v0.60.0 (2025-03-28)

### ✨ Enhancements

- Allow multiple leaf nodes in `NestedPipeFunc` ([#696](https://github.com/pipefunc/pipefunc/pull/696))
- Allow scoped `PipeFunc`s inside `NestedPipeFunc` ([#692](https://github.com/pipefunc/pipefunc/pull/692))
- Allow returning multiple outputs in `Pipeline.run` ([#694](https://github.com/pipefunc/pipefunc/pull/694))

### 🐛 Bug Fixes

- Fix `NestedPipeFunc.output_annotation` to handle renamed outputs with scopes ([#695](https://github.com/pipefunc/pipefunc/pull/695))

### 🧹 Maintenance

- Add `uv run` shebang to `get-notebooks.py` ([#691](https://github.com/pipefunc/pipefunc/pull/691))

### 🧪 Testing

- Skip flaky `test_pipeline_with_heterogeneous_executor` on nogil ([#690](https://github.com/pipefunc/pipefunc/pull/690))
- Skip test_parallel_memory_storage on 3.13 ([#689](https://github.com/pipefunc/pipefunc/pull/689))
- Actually test `NestedPipeFunc` with `SlurmExecutor` ([#687](https://github.com/pipefunc/pipefunc/pull/687))

### 📊 Stats

- `.md`: +13 lines, -23 lines
- `.py`: +345 lines, -62 lines
- `.toml`: +1 lines, -0 lines

## v0.59.1 (2025-03-19)

### 🐛 Bug Fixes

- Fix missing `resources_scope` in `NestedPipeFunc` ([#686](https://github.com/pipefunc/pipefunc/pull/686))

### 📚 Documentation

- Update `CHANGELOG.md` until v0.59.0 ([#684](https://github.com/pipefunc/pipefunc/pull/684))

### 📊 Stats

- `.py`: +50 lines, -4 lines
- `.md`: +113 lines, -10 lines

## v0.59.0 (2025-03-18)

### ✨ Enhancements

- Add `ResultDict.to_dataframe()` ([#681](https://github.com/pipefunc/pipefunc/pull/681))

### 🧹 Maintenance

- Pin bokeh<3.7 ([#682](https://github.com/pipefunc/pipefunc/pull/682))

### 🤖 CI

- Use `uv build` for PyPI releases ([#680](https://github.com/pipefunc/pipefunc/pull/680))

### 📊 Stats

- `.yml`: +5 lines, -12 lines
- `.py`: +80 lines, -2 lines
- `.toml`: +1 lines, -1 lines

## v0.58.1 (2025-03-11)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#676](https://github.com/pipefunc/pipefunc/pull/676))

### 📦 Dependencies

- ⬆️ Pin python to 3.13.2 ([#677](https://github.com/pipefunc/pipefunc/pull/677))

### 🧪 Testing

- Speedup tests ([#675](https://github.com/pipefunc/pipefunc/pull/675))

### ✨ Enhancements

- Select default initially in `VariantPipeline` selection widget ([#674](https://github.com/pipefunc/pipefunc/pull/674))

### 📊 Stats

- `.yml`: +2 lines, -2 lines
- `.yaml`: +2 lines, -2 lines
- `.py`: +128 lines, -23 lines

## v0.58.0 (2025-03-06)

### 🧪 Testing

- Add `scheduling_strategy="eager"` benchmarks ([#669](https://github.com/pipefunc/pipefunc/pull/669))

### ✨ Enhancements

- Allow multiple variant groups per `PipeFunc` (breaking change) ([#673](https://github.com/pipefunc/pipefunc/pull/673))

### 📊 Stats

- `.md`: +11 lines, -11 lines
- `.py`: +414 lines, -171 lines

## v0.57.2 (2025-03-05)

### 🐛 Bug Fixes

- Prefix the SLURM job names with `executor.name` ([#671](https://github.com/pipefunc/pipefunc/pull/671))

### 📊 Stats

- `.py`: +9 lines, -4 lines

## v0.57.1 (2025-03-05)

### 🐛 Bug Fixes

- Fix progress bar for `chunksize>1` ([#668](https://github.com/pipefunc/pipefunc/pull/668))
- Fix deepcopy/serialization for `Pipeline` with shared cache ([#657](https://github.com/pipefunc/pipefunc/pull/657))
- FIX `ResultDict.to_xarray` when no inputs ([#667](https://github.com/pipefunc/pipefunc/pull/667))
- Prevent ZeroDivisionError in auto-chunks ([#666](https://github.com/pipefunc/pipefunc/pull/666))

### 🧪 Testing

- Add `auto_subpipeline` test ([#661](https://github.com/pipefunc/pipefunc/pull/661))

### 📚 Documentation

- Update `CHANGELOG.md` until v0.57.0 ([#664](https://github.com/pipefunc/pipefunc/pull/664))

### 📊 Stats

- `.md`: +60 lines, -1 lines
- `.py`: +296 lines, -38 lines

## v0.57.0 (2025-03-04)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#660](https://github.com/pipefunc/pipefunc/pull/660))

### ✨ Enhancements

- Add `ResultDict.to_xarray` ([#656](https://github.com/pipefunc/pipefunc/pull/656))
- Add `pipeline.map_async(..., scheduling_strategy="eager")` ([#662](https://github.com/pipefunc/pipefunc/pull/662))
- Eager execution of graph, adds `pipeline.map(..., scheduling_strategy="eager")` ([#659](https://github.com/pipefunc/pipefunc/pull/659))

### 📊 Stats

- `.py`: +1527 lines, -15 lines
- `.yaml`: +1 lines, -1 lines

## v0.56.0 (2025-03-01)

### ✨ Enhancements

- Add option to `pipeline.map(..., return_results=False)` ([#626](https://github.com/pipefunc/pipefunc/pull/626))
- Simplify dynamic shape setting (do not rely on `result_array`) ([#652](https://github.com/pipefunc/pipefunc/pull/652))

### 🐛 Bug Fixes

- Fix `DictArray` with `internal_shape` who's entries have an additional dimension ([#654](https://github.com/pipefunc/pipefunc/pull/654))
- Fix persist and load `DictArray` and `SharedMemoryDictArray` ([#653](https://github.com/pipefunc/pipefunc/pull/653))

### 📊 Stats

- `.py`: +527 lines, -145 lines

## v0.55.2 (2025-02-24)

### 🐛 Bug Fixes

- Deal with defaults in the CLI that are not set ([#650](https://github.com/pipefunc/pipefunc/pull/650))

### 📊 Stats

- `.md`: +6 lines, -0 lines
- `.py`: +16 lines, -2 lines

## v0.55.1 (2025-02-24)

### 🐛 Bug Fixes

- Fix CLI with None default ([#649](https://github.com/pipefunc/pipefunc/pull/649))

### 📚 Documentation

- Update `CHANGELOG.md` until v0.55.0 ([#648](https://github.com/pipefunc/pipefunc/pull/648))

### 📊 Stats

- `.py`: +80 lines, -34 lines
- `.md`: +56 lines, -0 lines

## v0.55.0 (2025-02-24)

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.6.2 ([#633](https://github.com/pipefunc/pipefunc/pull/633))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#646](https://github.com/pipefunc/pipefunc/pull/646))

### 🧹 Maintenance

- Add rich-argparse to extras ([#645](https://github.com/pipefunc/pipefunc/pull/645))

### 📚 Documentation

- Fix Xarray CSS in dark mode for sphinx-book-theme ([#372](https://github.com/pipefunc/pipefunc/pull/372))
- Add real outputs to the CLI docs ([#647](https://github.com/pipefunc/pipefunc/pull/647))
- Add CLI concepts page ([#642](https://github.com/pipefunc/pipefunc/pull/642))

### ✨ Enhancements

- Add `docs` subcommand to CLI ([#644](https://github.com/pipefunc/pipefunc/pull/644))
- Add `Pipeline.cli()` that automatically generates a CLI ([#607](https://github.com/pipefunc/pipefunc/pull/607))
- Add `Pipeline.pydantic_model` ([#609](https://github.com/pipefunc/pipefunc/pull/609))

### 🐛 Bug Fixes

- Extract type annotation for classmethod ([#641](https://github.com/pipefunc/pipefunc/pull/641))

### 📊 Stats

- `.json`: +1 lines, -1 lines
- `.py`: +1090 lines, -15 lines
- `.yaml`: +1 lines, -1 lines
- `.yml`: +12 lines, -8 lines
- `.css`: +15 lines, -0 lines
- `.md`: +261 lines, -0 lines
- `.toml`: +6 lines, -4 lines
- `other`: +1 lines, -1 lines

## v0.54.1 (2025-02-17)

### ✨ Enhancements

- Add support for `SlurmExecutor` in `get_ncores` ([#640](https://github.com/pipefunc/pipefunc/pull/640))

### 📚 Documentation

- Update CHANGELOG.md until v0.54.0 ([#639](https://github.com/pipefunc/pipefunc/pull/639))

### 📊 Stats

- `.md`: +70 lines, -0 lines
- `.py`: +7 lines, -0 lines

## v0.54.0 (2025-02-17)

### 🐛 Bug Fixes

- Fix correct number of SLURM jobs for both `resources_scope` options ([#638](https://github.com/pipefunc/pipefunc/pull/638))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#637](https://github.com/pipefunc/pipefunc/pull/637))

### ✨ Enhancements

- Allow setting permissions in `DiskCache` ([#636](https://github.com/pipefunc/pipefunc/pull/636))

### 📊 Stats

- `.yaml`: +2 lines, -2 lines
- `.py`: +114 lines, -8 lines

## v0.53.3 (2025-02-06)

### 🐛 Bug Fixes

- Fix data loading with dynamic shapes ([#635](https://github.com/pipefunc/pipefunc/pull/635))

### 📊 Stats

- `.py`: +20 lines, -10 lines

## v0.53.2 (2025-02-05)

### 🐛 Bug Fixes

- Fix ND mapspec with multiple outputs and `internal_shape` ([#634](https://github.com/pipefunc/pipefunc/pull/634))

### 📊 Stats

- `.py`: +30 lines, -2 lines

## v0.53.1 (2025-02-05)

### Closed Issues

- Stateful callable output caching ([#510](https://github.com/pipefunc/pipefunc/issues/510))

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.27 ([#564](https://github.com/pipefunc/pipefunc/pull/564))

### ✨ Enhancements

- Raise more informative error when unknown variant selected ([#632](https://github.com/pipefunc/pipefunc/pull/632))

### 📚 Documentation

- Enable Plausible analytics ([#631](https://github.com/pipefunc/pipefunc/pull/631))
- Explain alternative SLURM method ([#630](https://github.com/pipefunc/pipefunc/pull/630))
- Add SLURM tutorial ([#629](https://github.com/pipefunc/pipefunc/pull/629))
- Fix links in documentation ([#625](https://github.com/pipefunc/pipefunc/pull/625))
- Update CHANGELOG.md until v0.53.0 ([#624](https://github.com/pipefunc/pipefunc/pull/624))
- Add page about caching ([#623](https://github.com/pipefunc/pipefunc/pull/623))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#628](https://github.com/pipefunc/pipefunc/pull/628))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.md`: +736 lines, -81 lines
- `.py`: +12 lines, -4 lines
- `.ipynb`: +1 lines, -1 lines
- `other`: +1 lines, -1 lines

## v0.53.0 (2025-01-31)

### ✨ Enhancements

- Include `__pipefunc_hash__` of function to determine the cache key ([#515](https://github.com/pipefunc/pipefunc/pull/515))
- Allow custom class with `__call__` ([#619](https://github.com/pipefunc/pipefunc/pull/619))
- Implement `NestedPipeFunc.parameter_annotations` ([#621](https://github.com/pipefunc/pipefunc/pull/621))
- Implement `Pipeline.parameter_annotations` and `Pipeline.output_annotations` ([#622](https://github.com/pipefunc/pipefunc/pull/622))

### 🐛 Bug Fixes

- Fix defaults and positional args ([#620](https://github.com/pipefunc/pipefunc/pull/620))

### 🧹 Maintenance

- Set `zarr>=2,<3` in `[extras]` ([#618](https://github.com/pipefunc/pipefunc/pull/618))

### 📊 Stats

- `.py`: +195 lines, -32 lines
- `.toml`: +1 lines, -1 lines

## v0.52.1 (2025-01-30)

### 🐛 Bug Fixes

- Fix pipefunc import with Zarr v3 (which is currently incompatible) ([#617](https://github.com/pipefunc/pipefunc/pull/617))

### 📊 Stats

- `.py`: +5 lines, -1 lines

## v0.52.0 (2025-01-30)

### ✨ Enhancements

- Make `Pipeline.validate()` public ([#616](https://github.com/pipefunc/pipefunc/pull/616))
- Add `__repr__` to Storage classes ([#614](https://github.com/pipefunc/pipefunc/pull/614))

### 📚 Documentation

- Update release notes up to v0.51.4 ([#613](https://github.com/pipefunc/pipefunc/pull/613))

### 📊 Stats

- `.md`: +60 lines, -0 lines
- `.py`: +53 lines, -10 lines

## v0.51.4 (2025-01-28)

### 🐛 Bug Fixes

- Fix unresolved shape in ≥2D arrays ([#612](https://github.com/pipefunc/pipefunc/pull/612))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#608](https://github.com/pipefunc/pipefunc/pull/608))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.py`: +87 lines, -5 lines

## v0.51.3 (2025-01-27)

### ✨ Enhancements

- Raise an appropriate error when returning different shapes with `internal_shapes` ([#610](https://github.com/pipefunc/pipefunc/pull/610))

### 🐛 Bug Fixes

- Fix 1 sized `internal_shapes` ([#611](https://github.com/pipefunc/pipefunc/pull/611))

### 📊 Stats

- `.py`: +58 lines, -5 lines

## v0.51.2 (2025-01-26)

### 🧹 Maintenance

- Fix dependency name in `[extras]` (`matplotlib-base` is conda name) ([#606](https://github.com/pipefunc/pipefunc/pull/606))

### 📚 Documentation

- Fix URLs linking to examples ([#604](https://github.com/pipefunc/pipefunc/pull/604))

### 📊 Stats

- `.py`: +1 lines, -1 lines
- `.ipynb`: +12 lines, -7 lines
- `.toml`: +1 lines, -1 lines

## v0.51.1 (2025-01-25)

### 🧹 Maintenance

- Rename `[extra]` optional dependencies to `[extras]` to align with `pipefunc-extras` ([#603](https://github.com/pipefunc/pipefunc/pull/603))

### 📚 Documentation

- Update release notes up to 0.51.0 ([#602](https://github.com/pipefunc/pipefunc/pull/602))

### 📊 Stats

- `.md`: +51 lines, -0 lines
- `.toml`: +2 lines, -2 lines

## v0.51.0 (2025-01-24)

### ✨ Enhancements

- Make `Result` a `dataclass` to avoid confusion with tuples ([#601](https://github.com/pipefunc/pipefunc/pull/601))
- Return a `ResultDict` that limits `__repr__` length in `pipeline.map` ([#600](https://github.com/pipefunc/pipefunc/pull/600))

### 📊 Stats

- `.py`: +73 lines, -31 lines

## v0.50.4 (2025-01-24)

### 🐛 Bug Fixes

- Fix case with `SlurmExecutor.finalize()` but nothing was submitted ([#599](https://github.com/pipefunc/pipefunc/pull/599))

### ✨ Enhancements

- Make `AsyncMap` a `dataclass` instead of `NamedTuple` ([#598](https://github.com/pipefunc/pipefunc/pull/598))

### 🤖 CI

- Do not allow `FIX:` prefix ([#597](https://github.com/pipefunc/pipefunc/pull/597))

### 📊 Stats

- `.py`: +7 lines, -3 lines
- `.json`: +0 lines, -1 lines
- `.yml`: +2 lines, -1 lines

## v0.50.3 (2025-01-24)

### Closed Issues

- BUG: Profiling `Pipeline.map` only works with `parallel=False` ([#547](https://github.com/pipefunc/pipefunc/issues/547))

### 🐛 Bug Fixes

- Fix `ZeroDivisionError` in `ProgressBar` ([#596](https://github.com/pipefunc/pipefunc/pull/596))

### ✨ Enhancements

- Emit warning when `profile=True` and `parallel=True` ([#594](https://github.com/pipefunc/pipefunc/pull/594))
- Allow setting custom colors in GraphViz graphs ([#593](https://github.com/pipefunc/pipefunc/pull/593))

### 📚 Documentation

- Release notes for 0.50.2 ([#592](https://github.com/pipefunc/pipefunc/pull/592))

### 📊 Stats

- `.md`: +18 lines, -0 lines
- `.py`: +179 lines, -45 lines

## v0.50.2 (2025-01-23)

### Closed Issues

- DOC: uv tip in tutorial.md is incorrect ([#588](https://github.com/pipefunc/pipefunc/issues/588))

### 🧹 Maintenance

- Add `pipefunc[extra]` optional dependencies to match `pipefunc-extra` on conda-forge ([#591](https://github.com/pipefunc/pipefunc/pull/591))

### 📚 Documentation

- Fix URL of tutorial in uv tip ([#590](https://github.com/pipefunc/pipefunc/pull/590))
- Update release notes and improve generation script ([#589](https://github.com/pipefunc/pipefunc/pull/589))

### 📊 Stats

- `.py`: +38 lines, -8 lines
- `.md`: +29 lines, -1 lines
- `.ipynb`: +3 lines, -3 lines
- `.toml`: +19 lines, -2 lines

## v0.50.1 (2025-01-23)

### 🐛 Bug Fixes

- Fix `map` over iterable with internal shape to `xarray` ([#587](https://github.com/pipefunc/pipefunc/pull/587))

### 🤖 CI

- Add PR title checking workflow ([#586](https://github.com/pipefunc/pipefunc/pull/586))

### 📚 Documentation

- Add CHANGELOG as a page to the documentation ([#584](https://github.com/pipefunc/pipefunc/pull/584))
- Add example with `ErrorSnapshot` and `Pipeline` ([#585](https://github.com/pipefunc/pipefunc/pull/585))
- Automatically generate `CHANGELOG.md` ([#580](https://github.com/pipefunc/pipefunc/pull/580))
- Add `get-notebooks.py` to the docs ([#582](https://github.com/pipefunc/pipefunc/pull/582))

### 🧪 Testing

- Check that `info` for `NestedPipeFunc` has no absorbed intermediate outputs ([#583](https://github.com/pipefunc/pipefunc/pull/583))

### 📊 Stats

- `.py`: +405 lines, -0 lines
- `.json`: +27 lines, -0 lines
- `.yml`: +52 lines, -0 lines
- `.md`: +1932 lines, -1 lines

## v0.50.0 (2025-01-21)

### Closed Issues

- Automatically parse doc-strings to generate Pipeline docs ([#562](https://github.com/pipefunc/pipefunc/issues/562))
- Create freeze button for scroll action in visualize_widget ([#561](https://github.com/pipefunc/pipefunc/issues/561))
- Scoped pipelines cannot be nested ([#374](https://github.com/pipefunc/pipefunc/issues/374))

### 📚 Documentation

- Add `get-notebooks.py` script that downloads all notebooks and puts them in a folder ([#581](https://github.com/pipefunc/pipefunc/pull/581))
- Add Raises section to `Pipeline.update_scope`'s docstring ([#572](https://github.com/pipefunc/pipefunc/pull/572))

### ✨ Enhancements

- Automatically generate documentation for `Pipeline`s ([#563](https://github.com/pipefunc/pipefunc/pull/563))
- Add literals of common storage options to `map` and `map_async` annotations ([#575](https://github.com/pipefunc/pipefunc/pull/575))

### 🧪 Testing

- Include pydantic in the micromamba testing `environment.yaml` ([#579](https://github.com/pipefunc/pipefunc/pull/579))
- Fix `tests/test_plotting.py::test_plotting_widget` ([#576](https://github.com/pipefunc/pipefunc/pull/576))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#577](https://github.com/pipefunc/pipefunc/pull/577))

### 📦 Dependencies

- ⬆️ Update release-drafter/release-drafter action to v6.1.0 ([#574](https://github.com/pipefunc/pipefunc/pull/574))

### 📊 Stats

- `.py`: +1333 lines, -14 lines
- `.yaml`: +2 lines, -2 lines
- `.yml`: +8 lines, -0 lines
- `.md`: +73 lines, -3 lines
- `.toml`: +2 lines, -1 lines

## v0.49.6 (2025-01-17)

### ✨ Enhancements

- Raise an exception if scope was not added to anything ([#571](https://github.com/pipefunc/pipefunc/pull/571))

### 📊 Stats

- `.py`: +15 lines, -1 lines

## v0.49.5 (2025-01-17)

### 🐛 Bug Fixes

- Fix using `Pipeline.arg_combinations` to calculate `root_args` ([#570](https://github.com/pipefunc/pipefunc/pull/570))

### 📚 Documentation

- Fix admonition in example.ipynb ([#569](https://github.com/pipefunc/pipefunc/pull/569))
- Rename `uvtip` -> `try-notebook` and use in `example.ipynb` ([#568](https://github.com/pipefunc/pipefunc/pull/568))
- Use triple backticks around `uv` command ([#567](https://github.com/pipefunc/pipefunc/pull/567))
- Add custom `uvtip` directive ([#566](https://github.com/pipefunc/pipefunc/pull/566))
- Small fixes ([#555](https://github.com/pipefunc/pipefunc/pull/555))

### 📊 Stats

- `.md`: +70 lines, -85 lines
- `.py`: +54 lines, -15 lines
- `.ipynb`: +4 lines, -12 lines

## v0.49.4 (2025-01-15)

### 🐛 Bug Fixes

- Fix `bound` in `NestedPipeFunc` with `scope` and `map` ([#560](https://github.com/pipefunc/pipefunc/pull/560))

### 📚 Documentation

- Recommendations of order ([#559](https://github.com/pipefunc/pipefunc/pull/559))

### 📊 Stats

- `.md`: +7 lines, -1 lines
- `.ipynb`: +1 lines, -0 lines
- `.py`: +77 lines, -46 lines

## v0.49.3 (2025-01-15)

### 🐛 Bug Fixes

- Fix `bound` in `NestedPipeFunc` inside `Pipeline` ([#557](https://github.com/pipefunc/pipefunc/pull/557))

### 📊 Stats

- `.py`: +106 lines, -0 lines

## v0.49.2 (2025-01-14)

### Closed Issues

- NestedPipeFunction in graph show wrong datatype ([#487](https://github.com/pipefunc/pipefunc/issues/487))

### 📚 Documentation

- Fix propagating defaults in `NestedPipeFunc` ([#558](https://github.com/pipefunc/pipefunc/pull/558))
- Rename "Benchmarking" to "Overhead and Efficiency" ([#553](https://github.com/pipefunc/pipefunc/pull/553))
- Add `visualize()` to `basic-usage.md` ([#552](https://github.com/pipefunc/pipefunc/pull/552))
- Add `opennb` to all examples ([#551](https://github.com/pipefunc/pipefunc/pull/551))
- Separate out examples into pages ([#550](https://github.com/pipefunc/pipefunc/pull/550))
- Fix simple typo ([#549](https://github.com/pipefunc/pipefunc/pull/549))
- Mention `uv` and `opennb` early in tutorial ([#548](https://github.com/pipefunc/pipefunc/pull/548))
- Reoganize the docs into pages ([#545](https://github.com/pipefunc/pipefunc/pull/545))

### ✨ Enhancements

- Change the order in which keys appear in `pipeline.info` ([#554](https://github.com/pipefunc/pipefunc/pull/554))

### 📊 Stats

- `.md}`: +8 lines, -1 lines
- `.md`: +2581 lines, -1351 lines
- `.py`: +76 lines, -9 lines
- `.ipynb`: +17 lines, -851 lines

## v0.49.1 (2025-01-13)

### 🐛 Bug Fixes

- Fix `NestedPipeFunction` in graph show wrong datatype ([#546](https://github.com/pipefunc/pipefunc/pull/546))

### 📚 Documentation

- Add a page about `mapspec` ([#543](https://github.com/pipefunc/pipefunc/pull/543))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#544](https://github.com/pipefunc/pipefunc/pull/544))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.yml`: +1 lines, -0 lines
- `.py`: +44 lines, -15 lines
- `.md`: +442 lines, -0 lines
- `.toml`: +1 lines, -0 lines

## v0.49.0 (2025-01-13)

### ✨ Enhancements

- Add a widget for `VariantPipeline.visualize()` and `VariantPipeline._repr_mimebundle_` ([#539](https://github.com/pipefunc/pipefunc/pull/539))

### 📚 Documentation

- Add `NestedPipeFunc` section to example notebooks and move `simplified_pipeline` to FAQ ([#542](https://github.com/pipefunc/pipefunc/pull/542))
- Fix method name of `Pipeline.join` in example notebook ([#541](https://github.com/pipefunc/pipefunc/pull/541))

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.18 ([#538](https://github.com/pipefunc/pipefunc/pull/538))

### 📊 Stats

- `.md`: +85 lines, -0 lines
- `.ipynb`: +98 lines, -164 lines
- `.py`: +283 lines, -16 lines
- `other`: +1 lines, -1 lines

## v0.48.2 (2025-01-11)

### 🐛 Bug Fixes

- Add more `NestedPipeFunc` tests and fix multiple outputs issue with them ([#536](https://github.com/pipefunc/pipefunc/pull/536))

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.17 ([#535](https://github.com/pipefunc/pipefunc/pull/535))

### 🧪 Testing

- Add multiple outputs to benchmarks ([#537](https://github.com/pipefunc/pipefunc/pull/537))

### 📊 Stats

- `.py`: +378 lines, -4 lines
- `other`: +1 lines, -1 lines

## v0.48.1 (2025-01-10)

### Closed Issues

- Add pipeline variants ([#517](https://github.com/pipefunc/pipefunc/issues/517))

### 🐛 Bug Fixes

- Fix scope for `NestedPipeFunc` ([#534](https://github.com/pipefunc/pipefunc/pull/534))

### 🧹 Maintenance

- Extend `.gitignore` ([#533](https://github.com/pipefunc/pipefunc/pull/533))

### 📊 Stats

- `.py`: +75 lines, -5 lines
- `other`: +6 lines, -0 lines

## v0.48.0 (2025-01-10)

### Closed Issues

- allow setting names of `NestedPipeFunc` by hand ([#195](https://github.com/pipefunc/pipefunc/issues/195))

### ✨ Enhancements

- Add `VariantPipelines.from_pipelines` classmethod ([#526](https://github.com/pipefunc/pipefunc/pull/526))
- Allow setting `NestedPipeFunc(..., function_name="customname")` ([#532](https://github.com/pipefunc/pipefunc/pull/532))

### 📊 Stats

- `.md`: +2 lines, -0 lines
- `.py`: +269 lines, -8 lines

## v0.47.3 (2025-01-10)

### 🐛 Bug Fixes

- Fix `combine_mapspecs` in `NestedPipeFunc` ([#531](https://github.com/pipefunc/pipefunc/pull/531))

### 📊 Stats

- `.py`: +10 lines, -7 lines

## v0.47.2 (2025-01-10)

### 🐛 Bug Fixes

- Set `internal_shape` for `NestedPipeFunc` ([#530](https://github.com/pipefunc/pipefunc/pull/530))
- Fix error message about using `map_async` with Slurm ([#528](https://github.com/pipefunc/pipefunc/pull/528))
- Fix case where bound and default are set for same parameter ([#525](https://github.com/pipefunc/pipefunc/pull/525))

### 🤖 CI

- Set `timeout-minutes: 10` in pytest jobs to prevent stuck 6 hour jobs ([#529](https://github.com/pipefunc/pipefunc/pull/529))

### 📚 Documentation

- Fix FAQ `VariantPipeline` example ([#524](https://github.com/pipefunc/pipefunc/pull/524))

### 📊 Stats

- `.yml`: +2 lines, -0 lines
- `.md`: +2 lines, -3 lines
- `.py`: +75 lines, -2 lines

## v0.47.1 (2025-01-09)

### 📚 Documentation

- Add example with non-unique variant names across `PipeFunc`s ([#520](https://github.com/pipefunc/pipefunc/pull/520))

### 🧹 Maintenance

- Pin `zarr>=2,<3` ([#521](https://github.com/pipefunc/pipefunc/pull/521))

### 📊 Stats

- `.yml`: +2 lines, -2 lines
- `.md`: +7 lines, -1 lines
- `.toml`: +1 lines, -1 lines

## v0.47.0 (2025-01-09)

### Closed Issues

- Aggregating function outputs into a `dict`? ([#456](https://github.com/pipefunc/pipefunc/issues/456))

### ✨ Enhancements

- Add `VariantPipeline` that can generate multiple `Pipeline` variants ([#518](https://github.com/pipefunc/pipefunc/pull/518))
- Add auto-chunksize heuristic ([#505](https://github.com/pipefunc/pipefunc/pull/505))

### 📝 Other

- Use Python 3.13 in CI where possible ([#519](https://github.com/pipefunc/pipefunc/pull/519))
- Update Discord invite link README.md ([#509](https://github.com/pipefunc/pipefunc/pull/509))

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.16 ([#516](https://github.com/pipefunc/pipefunc/pull/516))
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.15 ([#514](https://github.com/pipefunc/pipefunc/pull/514))
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.14 ([#511](https://github.com/pipefunc/pipefunc/pull/511))
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.13 ([#508](https://github.com/pipefunc/pipefunc/pull/508))
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.12 ([#507](https://github.com/pipefunc/pipefunc/pull/507))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#512](https://github.com/pipefunc/pipefunc/pull/512))
- [pre-commit.ci] pre-commit autoupdate ([#504](https://github.com/pipefunc/pipefunc/pull/504))

### 📚 Documentation

- Update number of required dependencies in README.md ([#506](https://github.com/pipefunc/pipefunc/pull/506))
- Autoformat Markdown in FAQ and README ([#503](https://github.com/pipefunc/pipefunc/pull/503))
- Add comparison with Snakemake ([#501](https://github.com/pipefunc/pipefunc/pull/501))

### 🤖 CI

- Revert `pull_request_target:` ([#500](https://github.com/pipefunc/pipefunc/pull/500))

### 📊 Stats

- `.py`: +1196 lines, -49 lines
- `.yml`: +4 lines, -4 lines
- `.yaml`: +2 lines, -2 lines
- `.md`: +201 lines, -63 lines
- `other`: +1 lines, -1 lines

## v0.46.0 (2024-12-23)

### Closed Issues

- Proposal: Reduce Pipeline.map's IPC overhead with chunking ([#484](https://github.com/pipefunc/pipefunc/issues/484))

### 🧪 Testing

- Explicitly set reason in `skipif` ([#499](https://github.com/pipefunc/pipefunc/pull/499))
- Skip shared memory test in CI on nogil Python (3.13t) ([#498](https://github.com/pipefunc/pipefunc/pull/498))

### ✨ Enhancements

- Allow providing an int to `chunksizes` ([#497](https://github.com/pipefunc/pipefunc/pull/497))
- Add `chunksizes` argument to `Pipeline.map` and `Pipeline.map_async` ([#493](https://github.com/pipefunc/pipefunc/pull/493))

### 🤖 CI

- Revert `pull_request_target:` for CodSpeed ([#495](https://github.com/pipefunc/pipefunc/pull/495))
- Use `pull_request_target:` to trigger CI on fork ([#494](https://github.com/pipefunc/pipefunc/pull/494))

### 📚 Documentation

- Mention HPC vs cloud based running ([#492](https://github.com/pipefunc/pipefunc/pull/492))
- How is this different from Dask, AiiDA, Luigi, Prefect, Kedro, Apache Airflow, etc.? ([#491](https://github.com/pipefunc/pipefunc/pull/491))
- Add Discord shield ([#490](https://github.com/pipefunc/pipefunc/pull/490))

### 📊 Stats

- `.yml`: +2 lines, -2 lines
- `.md`: +62 lines, -1 lines
- `.py`: +172 lines, -8 lines

## v0.45.0 (2024-12-21)

### Closed Issues

- Add helpers.getattr ([#480](https://github.com/pipefunc/pipefunc/issues/480))

### ✨ Enhancements

- Add `size_per_learner` for `SlurmExecutor` ([#486](https://github.com/pipefunc/pipefunc/pull/486))
- Add `helpers.get_attribute_factory` ([#481](https://github.com/pipefunc/pipefunc/pull/481))

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v5 ([#483](https://github.com/pipefunc/pipefunc/pull/483))
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.11 ([#482](https://github.com/pipefunc/pipefunc/pull/482))

### 📚 Documentation

- Set Python version to 3.13 in README `opennb` example ([#479](https://github.com/pipefunc/pipefunc/pull/479))
- Fix header level of "Dynamic Output Shapes and `internal_shapes`" ([#478](https://github.com/pipefunc/pipefunc/pull/478))
- Small formatting fix in example in doc-string ([#477](https://github.com/pipefunc/pipefunc/pull/477))

### 📊 Stats

- `.yml`: +4 lines, -4 lines
- `.md`: +1 lines, -1 lines
- `.ipynb`: +1 lines, -1 lines
- `.py`: +205 lines, -8 lines
- `.toml`: +1 lines, -1 lines
- `other`: +1 lines, -1 lines

## v0.44.0 (2024-12-19)

### ✨ Enhancements

- Add `Pipeline._repr_mimebundle_` ([#476](https://github.com/pipefunc/pipefunc/pull/476))
- Allow printing rich-formatted table with `pipeline.info()` ([#475](https://github.com/pipefunc/pipefunc/pull/475))
- Automatically set `internal_shape=("?", ...)` ([#463](https://github.com/pipefunc/pipefunc/pull/463))
- Add a `.devcontainer` for VS Code based on `uv` ([#473](https://github.com/pipefunc/pipefunc/pull/473))

### 📚 Documentation

- Update documentation about dynamic `internal_shapes` ([#474](https://github.com/pipefunc/pipefunc/pull/474))

### 📊 Stats

- `.json`: +43 lines, -0 lines
- `.py`: +162 lines, -75 lines
- `.yml`: +4 lines, -0 lines
- `.ipynb`: +138 lines, -73 lines
- `.toml`: +2 lines, -1 lines
- `other`: +18 lines, -0 lines

## v0.43.0 (2024-12-19)

### Closed Issues

- Allow `internal_shapes` to be input names (str) with simple expressions ([#197](https://github.com/pipefunc/pipefunc/issues/197))

### ✨ Enhancements

- Enable `show_progress` when using dynamic shapes ([#471](https://github.com/pipefunc/pipefunc/pull/471))
- Automatically set `internal_shape` ([#448](https://github.com/pipefunc/pipefunc/pull/448))

### 📚 Documentation

- Add workaround for multiple returns with different sizes ([#470](https://github.com/pipefunc/pipefunc/pull/470))
- Add `opennb` tip ([#464](https://github.com/pipefunc/pipefunc/pull/464))

### 🐛 Bug Fixes

- Fix case where there is no size ([#467](https://github.com/pipefunc/pipefunc/pull/467))
- Ensure to resolve shapes for all arrays in `_update_array` and fix `internal_shape` calculation ([#469](https://github.com/pipefunc/pipefunc/pull/469))
- Fix case where first dim is "?" ([#466](https://github.com/pipefunc/pipefunc/pull/466))
- Fix autogenerated `mapspec` issue with mismatching dims check ([#465](https://github.com/pipefunc/pipefunc/pull/465))

### 📊 Stats

- `.json`: +21 lines, -0 lines
- `.md`: +93 lines, -1 lines
- `.py`: +868 lines, -140 lines
- `.toml`: +2 lines, -3 lines

## v0.42.1 (2024-12-17)

### 🧪 Testing

- Use `pytest-timeout` plugin to prevent handing tests ([#459](https://github.com/pipefunc/pipefunc/pull/459))

### ✨ Enhancements

- Add `Pipeline.info()` that returns input and output info ([#462](https://github.com/pipefunc/pipefunc/pull/462))

### 📊 Stats

- `.yml`: +4 lines, -3 lines
- `.py`: +47 lines, -0 lines
- `.toml`: +3 lines, -1 lines

## v0.42.0 (2024-12-16)

### ✨ Enhancements

- Add `pipefunc.helpers.collect_kwargs` helper function ([#457](https://github.com/pipefunc/pipefunc/pull/457))
- Allow `pipeline.root_args(None)` (default) that returns all inputs ([#461](https://github.com/pipefunc/pipefunc/pull/461))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#460](https://github.com/pipefunc/pipefunc/pull/460))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.md`: +45 lines, -0 lines
- `.py`: +134 lines, -3 lines

## v0.41.3 (2024-12-13)

### 🐛 Bug Fixes

- Only use the cache when `cache=True` instead of always in `pipeline.map` ([#458](https://github.com/pipefunc/pipefunc/pull/458))

### 📊 Stats

- `.py`: +12 lines, -6 lines

## v0.41.2 (2024-12-11)

### 🐛 Bug Fixes

- Fix `internal_shapes` coming from `PipeFunc` constructor and `cleanup=False` ([#455](https://github.com/pipefunc/pipefunc/pull/455))

### 📊 Stats

- `.py`: +31 lines, -1 lines

## v0.41.1 (2024-12-11)

### Closed Issues

- Callback on each transition of a good way to visualize the result of each step beyond text ([#393](https://github.com/pipefunc/pipefunc/issues/393))
- Allow per `PipeFunc` storage ([#320](https://github.com/pipefunc/pipefunc/issues/320))
- Allow per `PipeFunc` executor (to mix parallel and local) ([#319](https://github.com/pipefunc/pipefunc/issues/319))
- `TypeError: 'NoneType' object cannot be interpreted as an integer` in documentation build ([#317](https://github.com/pipefunc/pipefunc/issues/317))
- ascii art ([#307](https://github.com/pipefunc/pipefunc/issues/307))

### 🐛 Bug Fixes

- Fix case with multiple output then iterate over single axis ([#454](https://github.com/pipefunc/pipefunc/pull/454))

### 🧹 Maintenance

- Small formatting changes and tiny refactors (from #448) ([#453](https://github.com/pipefunc/pipefunc/pull/453))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#452](https://github.com/pipefunc/pipefunc/pull/452))
- [pre-commit.ci] pre-commit autoupdate ([#445](https://github.com/pipefunc/pipefunc/pull/445))

### 📝 Other

- Add .ruff_cache to .gitignore ([#449](https://github.com/pipefunc/pipefunc/pull/449))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.py`: +143 lines, -40 lines
- `other`: +3 lines, -0 lines

## v0.41.0 (2024-11-27)

### ✨ Enhancements

- Add `post_execution_hook` for `PipeFunc` ([#306](https://github.com/pipefunc/pipefunc/pull/306))

### 📚 Documentation

- Set default plotting backend in docs to graphviz ([#441](https://github.com/pipefunc/pipefunc/pull/441))

### 📊 Stats

- `.md`: +50 lines, -0 lines
- `.ipynb`: +3 lines, -3 lines
- `.py`: +94 lines, -11 lines

## v0.40.2 (2024-11-27)

### 🧹 Maintenance

- Add `pipefunc[all]` to docs extras and remove `pydantic` from `[all]` ([#440](https://github.com/pipefunc/pipefunc/pull/440))
- Fix typo ([#439](https://github.com/pipefunc/pipefunc/pull/439))

### 📊 Stats

- `.py`: +11 lines, -4 lines
- `.yml`: +0 lines, -4 lines
- `.toml`: +2 lines, -1 lines

## v0.40.1 (2024-11-27)

### ✨ Enhancements

- Use `hatch` instead of `setuptools` ([#438](https://github.com/pipefunc/pipefunc/pull/438))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#429](https://github.com/pipefunc/pipefunc/pull/429))

### 📚 Documentation

- Fix admonition in example notebook ([#437](https://github.com/pipefunc/pipefunc/pull/437))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.ipynb`: +0 lines, -1 lines
- `.py`: +18 lines, -20 lines
- `.toml`: +13 lines, -16 lines

## v0.40.0 (2024-11-26)

### Closed Issues

- IndexError when handling exceptions without arguments in python<=3.11 ([#430](https://github.com/pipefunc/pipefunc/issues/430))

### 📚 Documentation

- Add a dropdown with interactive widget explanation ([#436](https://github.com/pipefunc/pipefunc/pull/436))

### ✨ Enhancements

- Add interactive version of `visualize_graphviz` ([#326](https://github.com/pipefunc/pipefunc/pull/326))
- Remove pygraphviz dependency, was only used in matplotlib plotting backend ([#433](https://github.com/pipefunc/pipefunc/pull/433))

### 🧹 Maintenance

- Sort the dependencies alphabetically ([#435](https://github.com/pipefunc/pipefunc/pull/435))

### 🤖 CI

- Test with plotting in uv now that `pygraphviz` is no longer required ([#434](https://github.com/pipefunc/pipefunc/pull/434))

### 🧪 Testing

- Fix pygraphviz <-> python-graphviz mixup in tests ([#432](https://github.com/pipefunc/pipefunc/pull/432))

### 📊 Stats

- `.yml`: +35 lines, -36 lines
- `.ipynb`: +137 lines, -122 lines
- `.py`: +103 lines, -12 lines
- `.toml`: +16 lines, -16 lines

## v0.39.0 (2024-11-26)

### 🐛 Bug Fixes

- Fix Python≤3.11 case for `handle_error` ([#431](https://github.com/pipefunc/pipefunc/pull/431))

### 🧹 Maintenance

- Install myst-nb with conda ([#428](https://github.com/pipefunc/pipefunc/pull/428))
- Remove `LazySequenceLearner` because of alternative in #381 ([#419](https://github.com/pipefunc/pipefunc/pull/419))

### ✨ Enhancements

- Avoid duplicate dependencies in .github/update-environment.py script ([#427](https://github.com/pipefunc/pipefunc/pull/427))
- Add support for `pydantic.BaseModel` ([#420](https://github.com/pipefunc/pipefunc/pull/420))
- Allow using memory based storages in parallel too ([#416](https://github.com/pipefunc/pipefunc/pull/416))

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v4 ([#426](https://github.com/pipefunc/pipefunc/pull/426))
- ⬆️ Update codecov/codecov-action action to v5 ([#424](https://github.com/pipefunc/pipefunc/pull/424))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#425](https://github.com/pipefunc/pipefunc/pull/425))
- [pre-commit.ci] pre-commit autoupdate ([#418](https://github.com/pipefunc/pipefunc/pull/418))

### 📚 Documentation

- Fix zarr API docs page ([#422](https://github.com/pipefunc/pipefunc/pull/422))
- Add section with `dataclass` and `pydantic.BaseModel` ([#421](https://github.com/pipefunc/pipefunc/pull/421))
- Add ultra-fast bullet point ([#417](https://github.com/pipefunc/pipefunc/pull/417))

### 📊 Stats

- `.py`: +233 lines, -204 lines
- `.yml`: +8 lines, -5 lines
- `.yaml`: +2 lines, -2 lines
- `.md`: +42 lines, -10 lines
- `.toml`: +3 lines, -1 lines

## v0.38.0 (2024-11-07)

### Closed Issues

- Dataclasses that use default_factory fields have buggy execution on second run ([#402](https://github.com/pipefunc/pipefunc/issues/402))
- Pipeline.add is not idempotent ([#394](https://github.com/pipefunc/pipefunc/issues/394))

### ✨ Enhancements

- Factor out `SlurmExecutor` logic from `_run.py` ([#415](https://github.com/pipefunc/pipefunc/pull/415))
- Rename _submit_single to _execute_single to avoid confusion with ex.submit ([#413](https://github.com/pipefunc/pipefunc/pull/413))
- Allow non-parallel progress bar ([#412](https://github.com/pipefunc/pipefunc/pull/412))
- Allow using `adaptive_scheduler.SlurmExecutor` ([#395](https://github.com/pipefunc/pipefunc/pull/395))
- Make `executor` a dict internally always ([#410](https://github.com/pipefunc/pipefunc/pull/410))
- Prevent duplicates from `PipeFunc`s that return multiple ([#409](https://github.com/pipefunc/pipefunc/pull/409))
- Add a `StoreType` ([#408](https://github.com/pipefunc/pipefunc/pull/408))
- Prevent adding functions with same `output_name` ([#404](https://github.com/pipefunc/pipefunc/pull/404))

### 🐛 Bug Fixes

- Also update progress bar for single executions ([#414](https://github.com/pipefunc/pipefunc/pull/414))

### 🧪 Testing

- Omit `pipefunc/map/_types.py` from coverage ([#411](https://github.com/pipefunc/pipefunc/pull/411))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#405](https://github.com/pipefunc/pipefunc/pull/405))

### 🧹 Maintenance

- Move `LazySequenceLearner` to separate module ([#407](https://github.com/pipefunc/pipefunc/pull/407))

### 📝 Other

- Define `ShapeDict`, `ShapeTuple`, `UserShapeDict` types ([#406](https://github.com/pipefunc/pipefunc/pull/406))

### 📊 Stats

- `.yaml`: +2 lines, -2 lines
- `.yml`: +2 lines, -2 lines
- `.ipynb`: +3 lines, -3 lines
- `.py`: +848 lines, -216 lines
- `.toml`: +4 lines, -2 lines

## v0.37.0 (2024-10-30)

### Closed Issues

- All values reported in profile_stats are 0 ([#392](https://github.com/pipefunc/pipefunc/issues/392))

### ✨ Enhancements

- Specially treat dataclasses with a default factory (closes #402) ([#403](https://github.com/pipefunc/pipefunc/pull/403))
- Update progress bar every second for first 30 seconds ([#401](https://github.com/pipefunc/pipefunc/pull/401))
- Include class name in `PipeFunc.__name__` ([#389](https://github.com/pipefunc/pipefunc/pull/389))
- Add `LazySequenceLearner` ([#385](https://github.com/pipefunc/pipefunc/pull/385))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#400](https://github.com/pipefunc/pipefunc/pull/400))
- [pre-commit.ci] pre-commit autoupdate ([#386](https://github.com/pipefunc/pipefunc/pull/386))

### 🧹 Maintenance

- Split up `_pipeline.py` into modules ([#399](https://github.com/pipefunc/pipefunc/pull/399))
- Use relative imports in `pipefunc.map` ([#398](https://github.com/pipefunc/pipefunc/pull/398))
- pipefunc.map module reorganization ([#397](https://github.com/pipefunc/pipefunc/pull/397))
- Move storage related modules to `map/_storage` ([#396](https://github.com/pipefunc/pipefunc/pull/396))

### 📚 Documentation

- Fix url in shield ([#391](https://github.com/pipefunc/pipefunc/pull/391))

### 🤖 CI

- Rename GitHub Actions workflows and test with minimal dependencies ([#390](https://github.com/pipefunc/pipefunc/pull/390))

### 📝 Other

- Add `uv` based GitHub Actions workflow and test on free-threaded Python 3.13t ([#387](https://github.com/pipefunc/pipefunc/pull/387))

### 🧪 Testing

- Make optional deps also optional in tests ([#388](https://github.com/pipefunc/pipefunc/pull/388))

### 📊 Stats

- `.py`: +1367 lines, -637 lines
- `.yml}`: +2 lines, -2 lines
- `.yml`: +53 lines, -5 lines
- `.yaml`: +2 lines, -2 lines
- `.md`: +1 lines, -1 lines
- `.py}`: +233 lines, -514 lines
- `.toml`: +3 lines, -5 lines
- `other`: +1 lines, -1 lines

## v0.36.1 (2024-10-17)

### 🧹 Maintenance

- Enable Python 3.13 in CI ([#384](https://github.com/pipefunc/pipefunc/pull/384))

### 🐛 Bug Fixes

- Use `internal_shapes` defined in `@pipefunc` in `create_learners` ([#383](https://github.com/pipefunc/pipefunc/pull/383))

### 📊 Stats

- `.yml`: +1 lines, -1 lines
- `.py`: +23 lines, -1 lines

## v0.36.0 (2024-10-16)

### 📝 Other

- Python 3.13 support ([#382](https://github.com/pipefunc/pipefunc/pull/382))

### 📚 Documentation

- Simplify example in README.md ([#379](https://github.com/pipefunc/pipefunc/pull/379))
- Add `html_theme_options` ([#371](https://github.com/pipefunc/pipefunc/pull/371))
- More improvements ([#370](https://github.com/pipefunc/pipefunc/pull/370))
- Reorder and reorganize docs ([#364](https://github.com/pipefunc/pipefunc/pull/364))
- Add `sphinx-notfound-page` for 404 ([#369](https://github.com/pipefunc/pipefunc/pull/369))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#377](https://github.com/pipefunc/pipefunc/pull/377))

### 📦 Dependencies

- ⬆️ Update mamba-org/setup-micromamba action to v2 ([#376](https://github.com/pipefunc/pipefunc/pull/376))

### 🧹 Maintenance

- Move `ProgressTracker` widget a `_widgets` folder ([#373](https://github.com/pipefunc/pipefunc/pull/373))

### 📊 Stats

- `.yml`: +9 lines, -8 lines
- `.yaml`: +2 lines, -2 lines
- `.md`: +200 lines, -35 lines
- `.py`: +26 lines, -10 lines
- `.ipynb`: +758 lines, -893 lines
- `.py}`: +0 lines, -0 lines
- `.toml`: +8 lines, -6 lines

## v0.35.1 (2024-09-30)

### ✨ Enhancements

- Allow pickling `DiskCache` without LRU Cache ([#368](https://github.com/pipefunc/pipefunc/pull/368))
- Allow `range(...)` as input in `map` ([#365](https://github.com/pipefunc/pipefunc/pull/365))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#366](https://github.com/pipefunc/pipefunc/pull/366))

### 📚 Documentation

- Use Ruff badge instead of Black ([#367](https://github.com/pipefunc/pipefunc/pull/367))
- Improve intro in README ([#363](https://github.com/pipefunc/pipefunc/pull/363))
- New title and tag line ([#362](https://github.com/pipefunc/pipefunc/pull/362))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.md`: +6 lines, -4 lines
- `.py`: +30 lines, -4 lines

## v0.35.0 (2024-09-27)

### 📚 Documentation

- Inline `mapspec` in physics based example ([#361](https://github.com/pipefunc/pipefunc/pull/361))
- Rely on latest release of MyST ([#360](https://github.com/pipefunc/pipefunc/pull/360))
- Add FAQ entry about mixing executors and storages ([#359](https://github.com/pipefunc/pipefunc/pull/359))
- Fix list formatting in Sphinx docs ([#358](https://github.com/pipefunc/pipefunc/pull/358))

### ✨ Enhancements

- Allow a different `Executor` per `PipeFunc` ([#357](https://github.com/pipefunc/pipefunc/pull/357))
- Allow setting a `storage` per `PipeFunc` ([#356](https://github.com/pipefunc/pipefunc/pull/356))
- Fallback to serialization for cache keys ([#355](https://github.com/pipefunc/pipefunc/pull/355))
- Set `fallback_to_str` to False by default for caching ([#354](https://github.com/pipefunc/pipefunc/pull/354))

### 📊 Stats

- `.py`: +701 lines, -132 lines
- `.yml`: +1 lines, -1 lines
- `.md`: +66 lines, -1 lines
- `.ipynb`: +4 lines, -10 lines
- `.toml`: +0 lines, -2 lines

## v0.34.0 (2024-09-25)

### ✨ Enhancements

- Add more space between `:` and name in `visualize_graphviz` ([#353](https://github.com/pipefunc/pipefunc/pull/353))
- Add `pipefunc.testing.patch` ([#352](https://github.com/pipefunc/pipefunc/pull/352))
- Include mapspec axis in the outputs of `PipeFunc` directly ([#349](https://github.com/pipefunc/pipefunc/pull/349))
- Keep mapspec in argument nodes in `visualize_graphviz` ([#348](https://github.com/pipefunc/pipefunc/pull/348))

### 📚 Documentation

- Add mapspec plots to tutorial ([#351](https://github.com/pipefunc/pipefunc/pull/351))

### 🧹 Maintenance

- Remove trailing commas to have arg lists on single line ([#350](https://github.com/pipefunc/pipefunc/pull/350))

### 📊 Stats

- `.md`: +49 lines, -0 lines
- `.ipynb`: +88 lines, -61 lines
- `.py`: +191 lines, -128 lines

## v0.33.0 (2024-09-24)

### ✨ Enhancements

- Add `pipeline.map_async` and a progress bar ([#333](https://github.com/pipefunc/pipefunc/pull/333))
- Raise an error with a helpful error message for missing dependencies ([#347](https://github.com/pipefunc/pipefunc/pull/347))
- Add optimized `FileArray.mask_linear` ([#346](https://github.com/pipefunc/pipefunc/pull/346))
- Refactor `pipeline.map.run` to prepare for async implementation ([#334](https://github.com/pipefunc/pipefunc/pull/334))
- Speedup code by 40% via simple change ([#337](https://github.com/pipefunc/pipefunc/pull/337))
- Improve missing plotting backend error message ([#332](https://github.com/pipefunc/pipefunc/pull/332))

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v3 ([#344](https://github.com/pipefunc/pipefunc/pull/344))

### 🤖 CI

- Remove unused steps from pytest pipeline ([#345](https://github.com/pipefunc/pipefunc/pull/345))

### 🧪 Testing

- Add a CI pipeline that checks for matching doc-strings ([#343](https://github.com/pipefunc/pipefunc/pull/343))
- Add benchmark from FAQ to test suite ([#338](https://github.com/pipefunc/pipefunc/pull/338))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#340](https://github.com/pipefunc/pipefunc/pull/340))

### 🐛 Bug Fixes

- Load custom objects correctly in `xarray` ([#336](https://github.com/pipefunc/pipefunc/pull/336))

### 📚 Documentation

- Add FAQ question about overhead/performance ([#335](https://github.com/pipefunc/pipefunc/pull/335))

### 📊 Stats

- `.py`: +1378 lines, -63 lines
- `.yml`: +44 lines, -9 lines
- `.yaml`: +1 lines, -1 lines
- `.md`: +53 lines, -0 lines
- `.toml`: +8 lines, -4 lines

## v0.32.1 (2024-09-18)

### 🐛 Bug Fixes

- Improve the parallel store compatibility checking function ([#331](https://github.com/pipefunc/pipefunc/pull/331))

### 📊 Stats

- `.py`: +73 lines, -8 lines

## v0.32.0 (2024-09-18)

### Closed Issues

- Add `pipefunc.map.Result.to_xarray` ([#312](https://github.com/pipefunc/pipefunc/issues/312))

### ✨ Enhancements

- Allow `pipeline.map` to run without disk ([#327](https://github.com/pipefunc/pipefunc/pull/327))
- Make Graphviz PipeFunc nodes rounded ([#329](https://github.com/pipefunc/pipefunc/pull/329))
- Implement `graphviz` based visualization ([#323](https://github.com/pipefunc/pipefunc/pull/323))
- Allow `visualize` to take an int for `figsize` (square) ([#322](https://github.com/pipefunc/pipefunc/pull/322))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#324](https://github.com/pipefunc/pipefunc/pull/324))

### 📚 Documentation

- Explain what a pipeline is ([#321](https://github.com/pipefunc/pipefunc/pull/321))
- Use `DiskCache` to prevent #317 ([#318](https://github.com/pipefunc/pipefunc/pull/318))

### 📊 Stats

- `.py`: +1128 lines, -311 lines
- `.yml`: +4 lines, -2 lines
- `.yaml`: +1 lines, -1 lines
- `.md`: +5 lines, -1 lines
- `.ipynb`: +175 lines, -122 lines
- `.toml`: +1 lines, -1 lines

## v0.31.1 (2024-09-11)

### 📚 Documentation

- Add a FAQ question about `ErrorSnapshot` and improve IP getting ([#316](https://github.com/pipefunc/pipefunc/pull/316))

### 📝 Other

- Note ([#315](https://github.com/pipefunc/pipefunc/pull/315))

### 📊 Stats

- `.md`: +49 lines, -0 lines
- `.ipynb`: +122 lines, -112 lines
- `.py`: +18 lines, -4 lines

## v0.31.0 (2024-09-10)

### ✨ Enhancements

- Add function going from `Results` to xarray with `xarray_dataset_from_results` ([#314](https://github.com/pipefunc/pipefunc/pull/314))
- Attach `ErrorSnapshot` for debugging ([#313](https://github.com/pipefunc/pipefunc/pull/313))
- Use pickle for cache key, inspired by `python-diskcache` package ([#310](https://github.com/pipefunc/pipefunc/pull/310))

### 📚 Documentation

- Add additional examples to the tutorial ([#311](https://github.com/pipefunc/pipefunc/pull/311))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#308](https://github.com/pipefunc/pipefunc/pull/308))

### 📝 Other

- Use `repr` for filename key ([#309](https://github.com/pipefunc/pipefunc/pull/309))
- TYP: Fix annotation of `output_picker` ([#303](https://github.com/pipefunc/pipefunc/pull/303))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.yml`: +3 lines, -0 lines
- `.ipynb`: +585 lines, -50 lines
- `.py`: +262 lines, -18 lines
- `.toml`: +16 lines, -2 lines

## v0.30.0 (2024-09-05)

### ✨ Enhancements

- Add `internal_shape` to `PipeFunc` ([#302](https://github.com/pipefunc/pipefunc/pull/302))

### 📚 Documentation

- Show triangulation on top of `Learner2D` plot ([#301](https://github.com/pipefunc/pipefunc/pull/301))

### 📊 Stats

- `.md`: +1 lines, -1 lines
- `.py`: +63 lines, -1 lines

## v0.29.0 (2024-09-05)

### Closed Issues

- Do type validation in pipeline definition ([#266](https://github.com/pipefunc/pipefunc/issues/266))
- Allow caching for `map` ([#264](https://github.com/pipefunc/pipefunc/issues/264))
- allow to inspect the resources inside the function ([#192](https://github.com/pipefunc/pipefunc/issues/192))
- allow internal parallelization ([#191](https://github.com/pipefunc/pipefunc/issues/191))

### ✨ Enhancements

- Add call to action ([#300](https://github.com/pipefunc/pipefunc/pull/300))
- Add ToC of questions to FAQ ([#298](https://github.com/pipefunc/pipefunc/pull/298))
- Add tl;dr note in API docs ([#297](https://github.com/pipefunc/pipefunc/pull/297))
- Skip parallelization if pointless ([#293](https://github.com/pipefunc/pipefunc/pull/293))
- Simpler example with `output_picker` ([#287](https://github.com/pipefunc/pipefunc/pull/287))

### 🐛 Bug Fixes

- Formatting in `is_object_array_type` doc-string ([#296](https://github.com/pipefunc/pipefunc/pull/296))
- formatting of lists in doc-strings ([#295](https://github.com/pipefunc/pipefunc/pull/295))
- doc-string of `func_dependents` and `func_dependencies` ([#294](https://github.com/pipefunc/pipefunc/pull/294))
- Correctly set cache value for `HybridCache` ([#292](https://github.com/pipefunc/pipefunc/pull/292))

### 📝 Other

- Allow to use cache for `Pipeline.map` ([#291](https://github.com/pipefunc/pipefunc/pull/291))
- Add `pipefunc.cache` and `pipefunc.typing` to the reference documentation ([#290](https://github.com/pipefunc/pipefunc/pull/290))
- Add `.cache` attribute to function using `@memoize` ([#288](https://github.com/pipefunc/pipefunc/pull/288))

### 📊 Stats

- `.md`: +36 lines, -0 lines
- `.ipynb`: +6 lines, -8 lines
- `.py`: +186 lines, -46 lines

## v0.28.0 (2024-09-03)

### 📝 Other

- Rename `pipefunc._cache` to `pipefunc.cache` ([#286](https://github.com/pipefunc/pipefunc/pull/286))
- Update `asciinema` recording ([#281](https://github.com/pipefunc/pipefunc/pull/281))
- Add asciinema recording ([#280](https://github.com/pipefunc/pipefunc/pull/280))
- Build `dirhtml` Sphinx docs instead of `html` ([#279](https://github.com/pipefunc/pipefunc/pull/279))

### ✨ Enhancements

- Small type annotation fix in `memoize` ([#285](https://github.com/pipefunc/pipefunc/pull/285))
- Improve caching and add a `memoize` decorator ([#283](https://github.com/pipefunc/pipefunc/pull/283))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#284](https://github.com/pipefunc/pipefunc/pull/284))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.yml`: +4 lines, -0 lines
- `.md`: +2 lines, -0 lines
- `.py`: +320 lines, -21 lines
- `.py}`: +194 lines, -1 lines

## v0.27.3 (2024-08-29)

### 🐛 Bug Fixes

- Case where reduction happens and output is unresolvable ([#278](https://github.com/pipefunc/pipefunc/pull/278))

### 📝 Other

- Add `py.typed` (PEP 561) ([#277](https://github.com/pipefunc/pipefunc/pull/277))

### 📊 Stats

- `.py`: +25 lines, -5 lines
- `.typed`: +0 lines, -0 lines
- `.toml`: +6 lines, -3 lines

## v0.27.2 (2024-08-29)

### 📝 Other

- Fix type annotation bug with autogenerated axis with internal shape ([#276](https://github.com/pipefunc/pipefunc/pull/276))

### 📊 Stats

- `.py`: +28 lines, -15 lines

## v0.27.1 (2024-08-29)

### 📝 Other

- Skip on `NoAnnotation` ([#275](https://github.com/pipefunc/pipefunc/pull/275))
- Add type annotation checking documentation ([#274](https://github.com/pipefunc/pipefunc/pull/274))
- Enforce one-to-one mapping for renames and improve validation error messages ([#273](https://github.com/pipefunc/pipefunc/pull/273))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#267](https://github.com/pipefunc/pipefunc/pull/267))

### 📊 Stats

- `.yaml`: +2 lines, -2 lines
- `.md`: +95 lines, -7 lines
- `.py`: +36 lines, -1 lines

## v0.27.0 (2024-08-28)

### 📝 Other

- Allow disabling type validation ([#271](https://github.com/pipefunc/pipefunc/pull/271))
- Allow types to be generics ([#269](https://github.com/pipefunc/pipefunc/pull/269))
- Ignore ARG001 ruff rule in tests ([#270](https://github.com/pipefunc/pipefunc/pull/270))
- Try getting type-hints instead of allowing to error out ([#268](https://github.com/pipefunc/pipefunc/pull/268))
- Add parameter and output annotations and validate them during `Pipeline` construction ([#6](https://github.com/pipefunc/pipefunc/pull/6))
- Simplify Adaptive Scheduler code ([#263](https://github.com/pipefunc/pipefunc/pull/263))
- Set Ruff Python version to 3.10 ([#262](https://github.com/pipefunc/pipefunc/pull/262))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#259](https://github.com/pipefunc/pipefunc/pull/259))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.md`: +5 lines, -3 lines
- `.ipynb`: +26 lines, -19 lines
- `.py`: +1199 lines, -100 lines
- `.toml`: +4 lines, -2 lines

## v0.26.0 (2024-08-22)

### 📝 Other

- Allow single job per element inside a `MapSpec` via `resources_scope` ([#260](https://github.com/pipefunc/pipefunc/pull/260))
- Return correct data in SequenceLearner when `return_output` ([#261](https://github.com/pipefunc/pipefunc/pull/261))
- Add `pipeline.run` adaptive tools ([#257](https://github.com/pipefunc/pipefunc/pull/257))
- Remove indentation level ([#255](https://github.com/pipefunc/pipefunc/pull/255))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#254](https://github.com/pipefunc/pipefunc/pull/254))
- [pre-commit.ci] pre-commit autoupdate ([#253](https://github.com/pipefunc/pipefunc/pull/253))
- [pre-commit.ci] pre-commit autoupdate ([#252](https://github.com/pipefunc/pipefunc/pull/252))
- [pre-commit.ci] pre-commit autoupdate ([#250](https://github.com/pipefunc/pipefunc/pull/250))

### 📦 Dependencies

- ⬆️ Update CodSpeedHQ/action action to v3 ([#251](https://github.com/pipefunc/pipefunc/pull/251))

### 📊 Stats

- `.yml`: +1 lines, -1 lines
- `.yaml`: +2 lines, -2 lines
- `.ipynb`: +2 lines, -2 lines
- `.py`: +451 lines, -51 lines

## v0.25.0 (2024-07-19)

### 📝 Other

- Add `parallelization_mode` option ([#249](https://github.com/pipefunc/pipefunc/pull/249))

### 📊 Stats

- `.yml`: +2 lines, -2 lines
- `.ipynb`: +16 lines, -7 lines
- `.py`: +97 lines, -15 lines
- `.toml`: +1 lines, -1 lines

## v0.24.0 (2024-07-18)

### Closed Issues

- AssertionError raised in the case of a function without inputs. ([#238](https://github.com/pipefunc/pipefunc/issues/238))

### 📝 Other

- Make Resources serializable ([#247](https://github.com/pipefunc/pipefunc/pull/247))
- Support delayed `Resources` in Adaptive Scheduler integration ([#234](https://github.com/pipefunc/pipefunc/pull/234))
- Rename `Resources` attributes `cpus`, `gpus`, `nodes`, `cpus_per_node`, `time` ([#245](https://github.com/pipefunc/pipefunc/pull/245))
- Split parts of `test_pipefunc.py` into several files ([#242](https://github.com/pipefunc/pipefunc/pull/242))
- Raise an exception when parameters and output_name overlaps ([#241](https://github.com/pipefunc/pipefunc/pull/241))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#246](https://github.com/pipefunc/pipefunc/pull/246))
- [pre-commit.ci] pre-commit autoupdate ([#244](https://github.com/pipefunc/pipefunc/pull/244))

### 📊 Stats

- `.yaml`: +2 lines, -2 lines
- `.md`: +15 lines, -15 lines
- `.ipynb`: +12 lines, -5 lines
- `.py`: +2132 lines, -1820 lines
- `.py}`: +0 lines, -0 lines

## v0.23.1 (2024-06-28)

### 📝 Other

- Allow parameterless functions in a Pipeline ([#240](https://github.com/pipefunc/pipefunc/pull/240))
- Allow passing `loss_function` to `to_adaptive_learner` ([#239](https://github.com/pipefunc/pipefunc/pull/239))

### 📊 Stats

- `.py`: +80 lines, -13 lines

## v0.23.0 (2024-06-27)

### 📝 Other

- Add a poor man's adaptive integration ([#237](https://github.com/pipefunc/pipefunc/pull/237))

### 📊 Stats

- `.md`: +237 lines, -0 lines
- `.py`: +183 lines, -2 lines

## v0.22.2 (2024-06-27)

### 📝 Other

- Disallow mapping over bound arguments and fix mapping over defaults ([#236](https://github.com/pipefunc/pipefunc/pull/236))

### 📊 Stats

- `.py`: +87 lines, -19 lines

## v0.22.1 (2024-06-27)

### 📝 Other

- Always call validate in `add` to ensure mapspec axes are autogenerated ([#235](https://github.com/pipefunc/pipefunc/pull/235))

### 📊 Stats

- `.py`: +28 lines, -3 lines

## v0.22.0 (2024-06-26)

### 📝 Other

- Get rid of `PipeFunc._default_resources` ([#232](https://github.com/pipefunc/pipefunc/pull/232))
- Allow bound arguments to be unhashable ([#233](https://github.com/pipefunc/pipefunc/pull/233))
- Allow `resources` to be delayed via a `Callable[[dict], Resources]` ([#219](https://github.com/pipefunc/pipefunc/pull/219))
- Add `resources_variable` in `PipeFunc`  ([#220](https://github.com/pipefunc/pipefunc/pull/220))
- Fix the Python version for Codspeed ([#231](https://github.com/pipefunc/pipefunc/pull/231))

### 📊 Stats

- `.yml`: +8 lines, -0 lines
- `.md`: +142 lines, -4 lines
- `.py`: +409 lines, -110 lines

## v0.21.0 (2024-06-24)

### Closed Issues

- Changing PipeFunc should trigger Pipeline internal cache reset ([#203](https://github.com/pipefunc/pipefunc/issues/203))

### 📝 Other

- Fix `dev` section in pyproject.toml `[project.optional-dependencies]` ([1d1a4a2](https://github.com/pipefunc/pipefunc/commit/1d1a4a2))
- Fix `PipeFunc` that share defaults ([#230](https://github.com/pipefunc/pipefunc/pull/230))
- Add Codspeed speedtest/benchmarking CI ([#229](https://github.com/pipefunc/pipefunc/pull/229))
- Add Renovate CI integration ([#221](https://github.com/pipefunc/pipefunc/pull/221))
- Combine resources with default_resources in `PipeFunc` object ([#214](https://github.com/pipefunc/pipefunc/pull/214))
- Simplify `Pipeline.copy` ([#217](https://github.com/pipefunc/pipefunc/pull/217))
- Remove `PipeFunc.__getattr__` and define `PipeFunc.__name__` ([#216](https://github.com/pipefunc/pipefunc/pull/216))
- Always create a copy when calling `Pipeline.add` ([#215](https://github.com/pipefunc/pipefunc/pull/215))
- Keep `default_resources` in `Pipeline` and rename `resources_report` to `print_profiling_stats` to avoid confusion ([#213](https://github.com/pipefunc/pipefunc/pull/213))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#228](https://github.com/pipefunc/pipefunc/pull/228))

### 📦 Dependencies

- ⬆️ Update release-drafter/release-drafter action to v6 ([#227](https://github.com/pipefunc/pipefunc/pull/227))
- ⬆️ Update github/codeql-action action to v3 ([#226](https://github.com/pipefunc/pipefunc/pull/226))
- ⬆️ Update actions/setup-python action to v5 ([#225](https://github.com/pipefunc/pipefunc/pull/225))
- ⬆️ Update release-drafter/release-drafter action to v5.25.0 ([#222](https://github.com/pipefunc/pipefunc/pull/222))
- ⬆️ Update actions/checkout action to v4 ([#223](https://github.com/pipefunc/pipefunc/pull/223))

### 📊 Stats

- `.json`: +35 lines, -0 lines
- `.yml`: +47 lines, -8 lines
- `.yaml`: +4 lines, -4 lines
- `.md`: +2 lines, -1 lines
- `.ipynb`: +2 lines, -2 lines
- `.py`: +397 lines, -144 lines
- `.py}`: +2 lines, -2 lines
- `.toml`: +2 lines, -1 lines

## v0.20.0 (2024-06-19)

### 📝 Other

- Remove specialized Adaptive code, and generalize `map` ([#212](https://github.com/pipefunc/pipefunc/pull/212))
- Remove `save_function` from `PipeFunc` and `delayed_callback` from `_LazyFunction` ([#211](https://github.com/pipefunc/pipefunc/pull/211))
- Remove the PipeFunc.set_profile method ([#210](https://github.com/pipefunc/pipefunc/pull/210))
- Factor out `_MockPipeline` ([#209](https://github.com/pipefunc/pipefunc/pull/209))
- Use frozen and slotted dataclasses where possible ([#208](https://github.com/pipefunc/pipefunc/pull/208))
- Keep a WeakRef to the Pipeline in each PipeFunc to reset Pipeline cache ([#207](https://github.com/pipefunc/pipefunc/pull/207))

### 📊 Stats

- `.py`: +159 lines, -349 lines

## v0.19.0 (2024-06-17)

### 📝 Other

- Introduce parameter namespacing via `scope`s ([#201](https://github.com/pipefunc/pipefunc/pull/201))
- Make sure all cells are executed to ensure working docs ([#206](https://github.com/pipefunc/pipefunc/pull/206))
- Create a copy of a `PipeFunc` in `Pipeline.add` ([#205](https://github.com/pipefunc/pipefunc/pull/205))

### 📊 Stats

- `.yml`: +1 lines, -0 lines
- `.py`: +694 lines, -109 lines
- `.md`: +196 lines, -57 lines
- `.toml`: +1 lines, -0 lines

## v0.18.1 (2024-06-14)

### Closed Issues

- Rename outputs too in `update_renames` ([#189](https://github.com/pipefunc/pipefunc/issues/189))

### 📝 Other

- Clear internal cache after renaming and re-defaulting ([#202](https://github.com/pipefunc/pipefunc/pull/202))

### 📊 Stats

- `.py`: +2 lines, -0 lines

## v0.18.0 (2024-06-13)

### Closed Issues

- include single results in xarray ([#188](https://github.com/pipefunc/pipefunc/issues/188))
- Rename mapspecs in update_renames ([#184](https://github.com/pipefunc/pipefunc/issues/184))

### 📝 Other

- Allow renaming `output_name` in `update_renames` ([#200](https://github.com/pipefunc/pipefunc/pull/200))
- Set `run_folder=None` by default ([#198](https://github.com/pipefunc/pipefunc/pull/198))
- Rename `MapSpec` in `update_renames` ([#196](https://github.com/pipefunc/pipefunc/pull/196))
- Add FAQ ([#187](https://github.com/pipefunc/pipefunc/pull/187))
- Include single results as 0D arrays in `xarray.Dataset` ([#190](https://github.com/pipefunc/pipefunc/pull/190))
- Extend `to_slurm_run` to return `adaptive_scheduler.RunManager` ([#186](https://github.com/pipefunc/pipefunc/pull/186))
- Add edge to `NestedPipeFunc` ([#183](https://github.com/pipefunc/pipefunc/pull/183))
- Update `example.ipynb` tutorial ([#182](https://github.com/pipefunc/pipefunc/pull/182))

### 📊 Stats

- `.md`: +322 lines, -3 lines
- `.yml`: +1 lines, -0 lines
- `.py`: +334 lines, -98 lines
- `.ipynb`: +499 lines, -246 lines
- `.toml`: +1 lines, -0 lines

## v0.17.0 (2024-06-11)

### 📝 Other

- Add remaining `Sweep` tests to reach 100% coverage on all code :tada: ([#181](https://github.com/pipefunc/pipefunc/pull/181))
- Remove superseded sweep functions: `get_precalculation_order` and `get_min_sweep_sets` ([#180](https://github.com/pipefunc/pipefunc/pull/180))
- Allow passing `update_from` to `update_renames` ([#179](https://github.com/pipefunc/pipefunc/pull/179))
- Fix regression introduced in #156 ([#178](https://github.com/pipefunc/pipefunc/pull/178))
- Reimplement `Pipeline.simplified_pipeline` using `NestedPipeFunc` ([#156](https://github.com/pipefunc/pipefunc/pull/156))
- Reach 100% testing coverage in `pipefunc/_pipeline.py` ([#177](https://github.com/pipefunc/pipefunc/pull/177))
- Increase testing coverage ([#176](https://github.com/pipefunc/pipefunc/pull/176))
- Fix typo and add more references ([21d63c1](https://github.com/pipefunc/pipefunc/commit/21d63c1))

### 📊 Stats

- `.ipynb`: +44 lines, -170 lines
- `.py`: +518 lines, -540 lines
- `.toml`: +1 lines, -1 lines

## v0.16.0 (2024-06-10)

### 📝 Other

- Add pipeline.update_rename to example ([b552324](https://github.com/pipefunc/pipefunc/commit/b552324))
- Add `Pipeline.update_renames` ([#175](https://github.com/pipefunc/pipefunc/pull/175))
- Allow to nest all ([#174](https://github.com/pipefunc/pipefunc/pull/174))
- Add `Pipeline.nest_funcs` ([#173](https://github.com/pipefunc/pipefunc/pull/173))
- Do not rely on hashing when checking defaults ([#172](https://github.com/pipefunc/pipefunc/pull/172))
- Deal with unhashable defaults ([#171](https://github.com/pipefunc/pipefunc/pull/171))
- Add sanity checks ([#170](https://github.com/pipefunc/pipefunc/pull/170))
- Add `Pipeline.update_defaults` ([#169](https://github.com/pipefunc/pipefunc/pull/169))
- Add `pipeline.join` and `pipeline1 | pipeline2` ([#168](https://github.com/pipefunc/pipefunc/pull/168))
- HoloViews plotting improvements ([#166](https://github.com/pipefunc/pipefunc/pull/166))

### 📊 Stats

- `.ipynb`: +107 lines, -71 lines
- `.py`: +387 lines, -29 lines

## v0.15.1 (2024-06-07)

### 📝 Other

- Do not add `MapSpec` axis for bound parameters ([#165](https://github.com/pipefunc/pipefunc/pull/165))

### 📊 Stats

- `.py`: +28 lines, -11 lines

## v0.15.0 (2024-06-07)

### Closed Issues

- class CombinedFunc(PipeFunc) to nest pipelines ([#138](https://github.com/pipefunc/pipefunc/issues/138))

### 📝 Other

- Make bound values actual node types in the graph ([#160](https://github.com/pipefunc/pipefunc/pull/160))
- Fix setting `__version__` during onbuild ([#164](https://github.com/pipefunc/pipefunc/pull/164))
- Pass through `internal_shapes` in `create_learners` ([#162](https://github.com/pipefunc/pipefunc/pull/162))
- Use `xarray.merge(... compat="override")` to deal with merging issues ([#161](https://github.com/pipefunc/pipefunc/pull/161))
- Add missing API docs file for `pipefunc.map.adaptive_scheduler` ([2861da2](https://github.com/pipefunc/pipefunc/commit/2861da2))
- Add Adaptive Scheduler integration ([#159](https://github.com/pipefunc/pipefunc/pull/159))
- Mention Xarray earlier in the docs ([db77f24](https://github.com/pipefunc/pipefunc/commit/db77f24))
- Make `resources` a module ([#158](https://github.com/pipefunc/pipefunc/pull/158))
- Disallow spaces in `Resources(memory)` ([aeb2d72](https://github.com/pipefunc/pipefunc/commit/aeb2d72))
- Implement `resources` specification ([#157](https://github.com/pipefunc/pipefunc/pull/157))

### 📊 Stats

- `.yml`: +2 lines, -0 lines
- `.md`: +18 lines, -0 lines
- `.ipynb`: +12 lines, -6 lines
- `.py`: +1476 lines, -135 lines
- `.toml`: +14 lines, -4 lines

## v0.14.0 (2024-06-04)

### 📝 Other

- Reorder functions, put public code at top of modules ([#155](https://github.com/pipefunc/pipefunc/pull/155))
- Add `NestedPipeFunc` ([#153](https://github.com/pipefunc/pipefunc/pull/153))
- Set author in documentation to PipeFunc Developers ([638b819](https://github.com/pipefunc/pipefunc/commit/638b819))
- Fix typo ([09cdb64](https://github.com/pipefunc/pipefunc/commit/09cdb64))
- Rename to `auto_subpipeline` ([#150](https://github.com/pipefunc/pipefunc/pull/150))
- Add option to pick the `output_name` and partial inputs when running `pipeline.map` ([#127](https://github.com/pipefunc/pipefunc/pull/127))
- Include `pipefunc.map.adaptive` integration in docs ([#149](https://github.com/pipefunc/pipefunc/pull/149))
- Validate inputs to `PipeFunc` ([#148](https://github.com/pipefunc/pipefunc/pull/148))
- Add `PipeFunc.update_bound` to allow fixed parameters ([#110](https://github.com/pipefunc/pipefunc/pull/110))

### 📊 Stats

- `.py`: +1150 lines, -466 lines
- `.md`: +1 lines, -1 lines
- `.yml`: +4 lines, -0 lines
- `.ipynb`: +67 lines, -3 lines
- `.toml`: +1 lines, -1 lines

## v0.13.0 (2024-06-02)

### 📝 Other

- Make `versioningit` an optional runtime dependency ([#144](https://github.com/pipefunc/pipefunc/pull/144))
- Set MyST in .github/update-environment.py ([#143](https://github.com/pipefunc/pipefunc/pull/143))
- Fix `pipeline.mapspecs_as_strings` statement (which is a property now) ([a9302d7](https://github.com/pipefunc/pipefunc/commit/a9302d7))
- Drop support for Python 3.8 and 3.9 ([#142](https://github.com/pipefunc/pipefunc/pull/142))
- Factor out simplify functions to simplify module ([#141](https://github.com/pipefunc/pipefunc/pull/141))
- Factor out `resources_report` ([#140](https://github.com/pipefunc/pipefunc/pull/140))
- Make more `cached_property`s ([#139](https://github.com/pipefunc/pipefunc/pull/139))
- Make `PipeFunc.renames` a property to avoid mutation ([#137](https://github.com/pipefunc/pipefunc/pull/137))
- Copy defaults in copy method ([#135](https://github.com/pipefunc/pipefunc/pull/135))
- Define many independent Adaptive learners for cross-products ([#136](https://github.com/pipefunc/pipefunc/pull/136))
- Implement `pipeline.map(... fixed_indices)` which computes the output only for selected indices ([#129](https://github.com/pipefunc/pipefunc/pull/129))
- Add `PipeFunc.update_renames` and `PipeFunc.update_defaults` ([#128](https://github.com/pipefunc/pipefunc/pull/128))
- Remove unused helper functions to join sets and find common items ([#134](https://github.com/pipefunc/pipefunc/pull/134))
- Make `pipefunc.lazy` a public module ([#133](https://github.com/pipefunc/pipefunc/pull/133))
- Cleanup `__init__.py` and make the `sweep` module public ([#132](https://github.com/pipefunc/pipefunc/pull/132))
- Use `sphinx-autodoc-typehints` ([#131](https://github.com/pipefunc/pipefunc/pull/131))
- Rename `map_parameters` to `mapspec_names` ([#130](https://github.com/pipefunc/pipefunc/pull/130))
- Validate inputs when calling `pipeline.map` ([#126](https://github.com/pipefunc/pipefunc/pull/126))
- Documentation MyST fixes and style changes ([#125](https://github.com/pipefunc/pipefunc/pull/125))
- Parallel docs changes ([#124](https://github.com/pipefunc/pipefunc/pull/124))
- Parallelize all functions in the same generation ([#123](https://github.com/pipefunc/pipefunc/pull/123))
- Factor out `RunInfo` to separate module ([#122](https://github.com/pipefunc/pipefunc/pull/122))
- Add `Pipeline.replace` ([#121](https://github.com/pipefunc/pipefunc/pull/121))
- add `join_overlapping_sets` and `common_in_sets` ([#120](https://github.com/pipefunc/pipefunc/pull/120))
- Allow setting new defaults in `PipeFunc` ([#111](https://github.com/pipefunc/pipefunc/pull/111))

### 📊 Stats

- `.py`: +1794 lines, -883 lines
- `.yml`: +15 lines, -17 lines
- `.md`: +20 lines, -2 lines
- `.ipynb`: +46 lines, -14 lines
- `.py}`: +32 lines, -20 lines
- `.toml`: +20 lines, -17 lines

## v0.12.0 (2024-05-30)

### 📝 Other

- Add custom parallelism section to the docs ([#119](https://github.com/pipefunc/pipefunc/pull/119))
- Add `SharedDictArray` ([#118](https://github.com/pipefunc/pipefunc/pull/118))
- Revert _SharedDictStore name change ([4f5d84a](https://github.com/pipefunc/pipefunc/commit/4f5d84a))
- Fix typo in test function name ([21c5757](https://github.com/pipefunc/pipefunc/commit/21c5757))
- Implement native `DictArray` ([#117](https://github.com/pipefunc/pipefunc/pull/117))
- Transfer to `github.com/pipefunc` org ([#116](https://github.com/pipefunc/pipefunc/pull/116))
- Add tests for `Pipeline.independent_axes_in_mapspecs` ([#115](https://github.com/pipefunc/pipefunc/pull/115))
- Functionality to identify independent axes in collection of `MapSpec`s ([#84](https://github.com/pipefunc/pipefunc/pull/84))
- Store `RunInfo` as JSON instead of cloudpickled bytes ([#109](https://github.com/pipefunc/pipefunc/pull/109))
- Allow passing any `concurrent.futures.Executor` ([#108](https://github.com/pipefunc/pipefunc/pull/108))
- Rename `ZarrArray` to `ZarrFileArray` ([#107](https://github.com/pipefunc/pipefunc/pull/107))
- Add mention of Xarray and Zarr ([c2f1092](https://github.com/pipefunc/pipefunc/commit/c2f1092))
- Add `ZarrMemory` and `ZarrSharedMemory` ([#106](https://github.com/pipefunc/pipefunc/pull/106))
- Fix headers in API docs ([edbff78](https://github.com/pipefunc/pipefunc/commit/edbff78))

### 📊 Stats

- `.md`: +11 lines, -10 lines
- `.py`: +961 lines, -191 lines
- `.ipynb`: +57 lines, -0 lines
- `.toml`: +1 lines, -1 lines

## v0.11.0 (2024-05-28)

### 📝 Other

- Pass through storage and return `map` result as `dict[str, Result]` ([#104](https://github.com/pipefunc/pipefunc/pull/104))
- Test all storage and remove custom `gzip` cloudpickle because `zarr` already compresses ([#103](https://github.com/pipefunc/pipefunc/pull/103))
- Add `zarr` integration ([#101](https://github.com/pipefunc/pipefunc/pull/101))
- Fix `map` run order ([#102](https://github.com/pipefunc/pipefunc/pull/102))
- Add `MapSpec` in `Pipeline.visualize` ([#72](https://github.com/pipefunc/pipefunc/pull/72))
- Mention where example is based on ([d606bd6](https://github.com/pipefunc/pipefunc/commit/d606bd6))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#105](https://github.com/pipefunc/pipefunc/pull/105))

### 📊 Stats

- `.yaml`: +1 lines, -1 lines
- `.yml`: +3 lines, -0 lines
- `.md`: +18 lines, -0 lines
- `.ipynb`: +97 lines, -75 lines
- `.py`: +1585 lines, -628 lines
- `.toml`: +13 lines, -3 lines

## v0.10.0 (2024-05-24)

### 📝 Other

- Add `xarray` integration ([#94](https://github.com/pipefunc/pipefunc/pull/94))
- Make sure to only evaluate a function once when possible ([#100](https://github.com/pipefunc/pipefunc/pull/100))
- Only create cache if functions have caching enabled ([#99](https://github.com/pipefunc/pipefunc/pull/99))
- Use sphinx-book-theme instead of furo ([#98](https://github.com/pipefunc/pipefunc/pull/98))
- Make `output_to_func` a `cached_property` and `RunInfo` a `dataclass`, and some renames ([#96](https://github.com/pipefunc/pipefunc/pull/96))
- Replace `tabulate` dependency by simple function ([#97](https://github.com/pipefunc/pipefunc/pull/97))

### 📊 Stats

- `.yaml`: +0 lines, -1 lines
- `.yml`: +4 lines, -3 lines
- `.py`: +797 lines, -286 lines
- `.ipynb`: +264 lines, -3 lines
- `.toml`: +5 lines, -4 lines

## v0.9.0 (2024-05-22)

### 📝 Other

- Add support for output arrays with internal structure and autogenerate MapSpecs ([#85](https://github.com/pipefunc/pipefunc/pull/85))
- Style changes ([#93](https://github.com/pipefunc/pipefunc/pull/93))
- Allow calling `add_mapspec_axis` on multiple parameters ([#92](https://github.com/pipefunc/pipefunc/pull/92))
- Rename `manual_shapes` to `internal_shapes` ([#91](https://github.com/pipefunc/pipefunc/pull/91))
- Fix bug and refactor `FileArray` ([#90](https://github.com/pipefunc/pipefunc/pull/90))
- Add `PipeFunc.copy()` and use it when creating `Pipeline` with tuples including `MapSpec`s ([#89](https://github.com/pipefunc/pipefunc/pull/89))
- Implement `FileArray` with internal structure ([#88](https://github.com/pipefunc/pipefunc/pull/88))
- `MapSpec` method changes and add `Pipeline.mapspec_axes` and `mapspec_dimensions` ([#86](https://github.com/pipefunc/pipefunc/pull/86))
- Rephrase doc-string ([6f633f3](https://github.com/pipefunc/pipefunc/commit/6f633f3))
- Add zipping axis test and doc-string ([#83](https://github.com/pipefunc/pipefunc/pull/83))
- Create a temporary `run_folder` if `None` and `README.md` improvements ([#82](https://github.com/pipefunc/pipefunc/pull/82))
- Remove fan-out/fan-in ([6839dbe](https://github.com/pipefunc/pipefunc/commit/6839dbe))
- Add `mapspecs` method, `sorted_functions` property, and rewrite intro in `README` ([#81](https://github.com/pipefunc/pipefunc/pull/81))
- Better error message in `Pipeline.run` ([#80](https://github.com/pipefunc/pipefunc/pull/80))
- Fix bug for `add_mapspec_axis` ([#79](https://github.com/pipefunc/pipefunc/pull/79))
- Add `Pipeline.add_mapspec_axis` for cross-products ([#78](https://github.com/pipefunc/pipefunc/pull/78))
- Create separate API docs per module ([#77](https://github.com/pipefunc/pipefunc/pull/77))
- Fix header in example.ipynb ([ac39689](https://github.com/pipefunc/pipefunc/commit/ac39689))
- Reorder the docs and small rewrite ([#76](https://github.com/pipefunc/pipefunc/pull/76))
- Add docs section about renames ([#75](https://github.com/pipefunc/pipefunc/pull/75))
- Dump to `FileArray` as soon as possible ([#74](https://github.com/pipefunc/pipefunc/pull/74))
- Fix typo in docs and cache improvements ([#73](https://github.com/pipefunc/pipefunc/pull/73))

### 📊 Stats

- `.md`: +69 lines, -35 lines
- `.ipynb`: +316 lines, -136 lines
- `.py`: +2172 lines, -605 lines

## v0.8.0 (2024-05-17)

### 📝 Other

- Increase coverage and fix Sweep bug ([#71](https://github.com/pipefunc/pipefunc/pull/71))
- Add verbose flag ([#70](https://github.com/pipefunc/pipefunc/pull/70))
- Remove `_update_wrapper` to make `dataclass`es pickleble ([#69](https://github.com/pipefunc/pipefunc/pull/69))
- Compare `RunInfo` to old saved `RunInfo` ([#68](https://github.com/pipefunc/pipefunc/pull/68))
- Add picklable `_MapWrapper` used in `create_learners_from_sweep` ([#67](https://github.com/pipefunc/pipefunc/pull/67))
- Add loading of data that already exists in `Pipeline.map` ([#66](https://github.com/pipefunc/pipefunc/pull/66))
- Rename `get_cache` to `_current_cache` ([#63](https://github.com/pipefunc/pipefunc/pull/63))
- Rename to `_run_pipeline` to `run` to align with `map` ([#64](https://github.com/pipefunc/pipefunc/pull/64))

### 📊 Stats

- `.py`: +544 lines, -57 lines

## v0.7.0 (2024-05-15)

### 📝 Other

- Add pipefunc.map.adaptive to API docs ([bb3084f](https://github.com/pipefunc/pipefunc/commit/bb3084f))
- Better `resource_report` and add add `Sweep` with `MapSpec` tools ([#62](https://github.com/pipefunc/pipefunc/pull/62))
- Add `pipefunc.map` to API docs ([9f27833](https://github.com/pipefunc/pipefunc/commit/9f27833))
- Use updated logo ([d1c32ea](https://github.com/pipefunc/pipefunc/commit/d1c32ea))
- Docs improvements ([#61](https://github.com/pipefunc/pipefunc/pull/61))
- Remove Jupyterlite configuration ([538b8f5](https://github.com/pipefunc/pipefunc/commit/538b8f5))
- Add Map-Reduce to features list ([29af3bf](https://github.com/pipefunc/pipefunc/commit/29af3bf))
- Add `Pipeline.map` docs and automatically parallelize `map` ([#59](https://github.com/pipefunc/pipefunc/pull/59))
- Various small improvements ([#58](https://github.com/pipefunc/pipefunc/pull/58))
- Style changes (100 character lines) ([#57](https://github.com/pipefunc/pipefunc/pull/57))

### 📊 Stats

- `.py`: +570 lines, -346 lines
- `.yaml`: +1 lines, -1 lines
- `.yml`: +3 lines, -18 lines
- `.md`: +20 lines, -5 lines
- `.ipynb`: +229 lines, -12 lines
- `.toml`: +6 lines, -3 lines
- `.cfg`: +0 lines, -5 lines
- `other`: +4 lines, -0 lines

## v0.6.0 (2024-05-15)

### 📝 Other

- Integrate `MapSpec`ed `Pipeline`s with Adaptive ([#56](https://github.com/pipefunc/pipefunc/pull/56))
- Add functionality to run `Pipeline`s with `MapSpec`s ([#55](https://github.com/pipefunc/pipefunc/pull/55))
- Refactor, improve, test, and integrate `MapSpec` into `Pipeline` ([#22](https://github.com/pipefunc/pipefunc/pull/22))
- Add `MapSpec` and `FileBasedObjectArray` `from aiida-dynamic-workflows` ([#54](https://github.com/pipefunc/pipefunc/pull/54))
- Improve utils, add topological_generations, and better error message ([#53](https://github.com/pipefunc/pipefunc/pull/53))
- Fix docs (jupyterlite) ([#52](https://github.com/pipefunc/pipefunc/pull/52))
- Take out arg_combination functions ([#51](https://github.com/pipefunc/pipefunc/pull/51))
- Take out methods and make functions and simplify code ([#50](https://github.com/pipefunc/pipefunc/pull/50))
- dump, load, Pipeline.defaults, Pipeline.copy, and style ([#49](https://github.com/pipefunc/pipefunc/pull/49))
- `construct_dag` fix and remove dead code ([#47](https://github.com/pipefunc/pipefunc/pull/47))
- Refactor `Pipeline._execute_pipeline` ([#44](https://github.com/pipefunc/pipefunc/pull/44))
- Switch around log message ([#45](https://github.com/pipefunc/pipefunc/pull/45))
- Add test `test_full_output_cache` ([#46](https://github.com/pipefunc/pipefunc/pull/46))
- Fix test_handle_error on MacOS ([#43](https://github.com/pipefunc/pipefunc/pull/43))
- Better error message ([#42](https://github.com/pipefunc/pipefunc/pull/42))
- Raise when unused parameters are provided ([#41](https://github.com/pipefunc/pipefunc/pull/41))
- Add pipeline.drop ([#40](https://github.com/pipefunc/pipefunc/pull/40))
- Rename PipelineFunction -> PipeFunc ([#39](https://github.com/pipefunc/pipefunc/pull/39))
- Several caching fixes ([#38](https://github.com/pipefunc/pipefunc/pull/38))
- Use codecov/codecov-action@v4 ([#36](https://github.com/pipefunc/pipefunc/pull/36))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#48](https://github.com/pipefunc/pipefunc/pull/48))
- [pre-commit.ci] pre-commit autoupdate ([#37](https://github.com/pipefunc/pipefunc/pull/37))

### 📊 Stats

- `.yml`: +9 lines, -4 lines
- `.yaml`: +1 lines, -1 lines
- `.md`: +1 lines, -1 lines
- `.ipynb`: +5 lines, -5 lines
- `.py`: +3265 lines, -522 lines
- `.toml`: +4 lines, -2 lines
- `other`: +21 lines, -0 lines

## v0.5.0 (2024-04-30)

### 📝 Other

- Make positional only ([#35](https://github.com/pipefunc/pipefunc/pull/35))
- Format line ([6c80ed4](https://github.com/pipefunc/pipefunc/commit/6c80ed4))
- Remove unused var T ([8e57f06](https://github.com/pipefunc/pipefunc/commit/8e57f06))
- Reorganize some definitions into modules ([#34](https://github.com/pipefunc/pipefunc/pull/34))
- Add a TaskGraph ([#33](https://github.com/pipefunc/pipefunc/pull/33))
- Fix cache argument in docs and fix pickling issues ([#32](https://github.com/pipefunc/pipefunc/pull/32))
- Add 3.12 to testing matrix ([#31](https://github.com/pipefunc/pipefunc/pull/31))
- Optimizations ([#30](https://github.com/pipefunc/pipefunc/pull/30))
- Rename cloudpickle parameter ([#29](https://github.com/pipefunc/pipefunc/pull/29))
- Allow lazy pipeline evaluation ([#26](https://github.com/pipefunc/pipefunc/pull/26))
- Add Cache ABC ([#28](https://github.com/pipefunc/pipefunc/pull/28))
- Cache improvement and rename ([#27](https://github.com/pipefunc/pipefunc/pull/27))
- Add `with_cloudpickle` to `HybridCache` ([#25](https://github.com/pipefunc/pipefunc/pull/25))
- Add `DiskCache` ([#24](https://github.com/pipefunc/pipefunc/pull/24))
- Add root_args method ([#23](https://github.com/pipefunc/pipefunc/pull/23))
- Add hype tag (AI) ([547c44d](https://github.com/pipefunc/pipefunc/commit/547c44d))
- Rewrite the intro in the README ([#21](https://github.com/pipefunc/pipefunc/pull/21))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#20](https://github.com/pipefunc/pipefunc/pull/20))
- [pre-commit.ci] pre-commit autoupdate ([#19](https://github.com/pipefunc/pipefunc/pull/19))
- [pre-commit.ci] pre-commit autoupdate ([#18](https://github.com/pipefunc/pipefunc/pull/18))
- [pre-commit.ci] pre-commit autoupdate ([#17](https://github.com/pipefunc/pipefunc/pull/17))

### 📊 Stats

- `.yml`: +1 lines, -1 lines
- `.yaml`: +3 lines, -3 lines
- `.md`: +12 lines, -19 lines
- `.py`: +2411 lines, -1478 lines
- `.ipynb`: +2 lines, -2 lines

## v0.4.0 (2024-03-11)

### 📝 Other

- Keep functions picklable ([5446653](https://github.com/pipefunc/pipefunc/commit/5446653))
- Use kwargs with derivers ([168b1c2](https://github.com/pipefunc/pipefunc/commit/168b1c2))
- Fix typo ([1aa0ee1](https://github.com/pipefunc/pipefunc/commit/1aa0ee1))
- Rename callables to derivers ([#16](https://github.com/pipefunc/pipefunc/pull/16))
- Do not overwrite keys that exist in the sweep ([#15](https://github.com/pipefunc/pipefunc/pull/15))
- Add `callables` and `product` to Sweep ([#14](https://github.com/pipefunc/pipefunc/pull/14))
- Use Pipeline.leaf_nodes instead of unique tip ([#12](https://github.com/pipefunc/pipefunc/pull/12))
- Call `update_wrapper` for correct signature ([#11](https://github.com/pipefunc/pipefunc/pull/11))

### 📊 Stats

- `.py`: +402 lines, -29 lines

## v0.3.0 (2024-03-08)

### 📝 Other

- Automatically set `output_name` if possible ([#10](https://github.com/pipefunc/pipefunc/pull/10))
- Unique colors for combinable and non-combinable nodes ([#9](https://github.com/pipefunc/pipefunc/pull/9))
- Fix coloring of combinable nodes ([#8](https://github.com/pipefunc/pipefunc/pull/8))
- Allow constants in `Sweep`s ([#7](https://github.com/pipefunc/pipefunc/pull/7))
- Allow constants in a Sweep ([#4](https://github.com/pipefunc/pipefunc/pull/4))
- Fix line length ([#5](https://github.com/pipefunc/pipefunc/pull/5))
- Color combinable and add test ([dfc44c0](https://github.com/pipefunc/pipefunc/commit/dfc44c0))
- remove cell ([0115807](https://github.com/pipefunc/pipefunc/commit/0115807))
- Rename reduce -> simplify ([ed4217a](https://github.com/pipefunc/pipefunc/commit/ed4217a))
- Remove incorrect copyright message ([a937aed](https://github.com/pipefunc/pipefunc/commit/a937aed))
- Update environment.yml ([88eb9ae](https://github.com/pipefunc/pipefunc/commit/88eb9ae))
- Skip plotting ([302f647](https://github.com/pipefunc/pipefunc/commit/302f647))
- Fix test dependencies (add pandas) ([b0d5c62](https://github.com/pipefunc/pipefunc/commit/b0d5c62))
- More pre-commit and typing fixes ([de022cd](https://github.com/pipefunc/pipefunc/commit/de022cd))
- Use ruff-format ([cea11f9](https://github.com/pipefunc/pipefunc/commit/cea11f9))
- Fix pre-commit issues ([a5d6f34](https://github.com/pipefunc/pipefunc/commit/a5d6f34))
- Update pre-commit filters ([238466b](https://github.com/pipefunc/pipefunc/commit/238466b))
- Fix pip install command in README.md ([5f6278e](https://github.com/pipefunc/pipefunc/commit/5f6278e))

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate ([#3](https://github.com/pipefunc/pipefunc/pull/3))

### 📊 Stats

- `.py`: +366 lines, -78 lines
- `.yaml`: +6 lines, -8 lines
- `.md`: +2 lines, -2 lines
- `.yml`: +2 lines, -0 lines
- `.ipynb`: +101 lines, -79 lines
- `.toml`: +10 lines, -7 lines

## v0.2.0 (2023-11-27)

### Closed Issues

- Header ([#1](https://github.com/pipefunc/pipefunc/issues/1))

### 📝 Other

- Add Python 3.12 classifier ([13091a9](https://github.com/pipefunc/pipefunc/commit/13091a9))
- Fix doc-string ([fc1b645](https://github.com/pipefunc/pipefunc/commit/fc1b645))
- Remove print statement ([7f64826](https://github.com/pipefunc/pipefunc/commit/7f64826))
- Install black[jupyter] in dev deps ([8669f08](https://github.com/pipefunc/pipefunc/commit/8669f08))
- Add saving ([8329ba5](https://github.com/pipefunc/pipefunc/commit/8329ba5))
- Unshallow clone in Readthedocs ([5d7ae83](https://github.com/pipefunc/pipefunc/commit/5d7ae83))
- Add shields ([1f09df6](https://github.com/pipefunc/pipefunc/commit/1f09df6))
- Fix type hint in _sweep.py ([d71ad78](https://github.com/pipefunc/pipefunc/commit/d71ad78))
- Another typo fix in all_transitive_paths ([8551a6a](https://github.com/pipefunc/pipefunc/commit/8551a6a))
- Fix type in doc-string ([18dcd2f](https://github.com/pipefunc/pipefunc/commit/18dcd2f))
- Add all_transitive_paths to get parallel and indepent computation chains ([2c0e8a9](https://github.com/pipefunc/pipefunc/commit/2c0e8a9))
- Add leaf and root nodes property ([f5ebacf](https://github.com/pipefunc/pipefunc/commit/f5ebacf))
- Fix _assert_valid_sweep_dict ([3485eec](https://github.com/pipefunc/pipefunc/commit/3485eec))
- Add get_min_sweep_sets ([a2d46d0](https://github.com/pipefunc/pipefunc/commit/a2d46d0))
- Rewrap text in doc-strings ([dc73186](https://github.com/pipefunc/pipefunc/commit/dc73186))
- Add all_execution_orders ([eaf71f5](https://github.com/pipefunc/pipefunc/commit/eaf71f5))
- Add conservatively_combine ([320b6ef](https://github.com/pipefunc/pipefunc/commit/320b6ef))
- rename 'add' to 'combine' ([558e23c](https://github.com/pipefunc/pipefunc/commit/558e23c))
- Remove [project.scripts] section from pyproject.toml ([0375761](https://github.com/pipefunc/pipefunc/commit/0375761))

### 📊 Stats

- `.yml`: +3 lines, -0 lines
- `.md`: +11 lines, -0 lines
- `.py`: +410 lines, -62 lines
- `.toml`: +2 lines, -4 lines

## v0.1.0 (2023-07-16)

### 📝 Other

- Fix license in pyproject.toml ([5bec316](https://github.com/pipefunc/pipefunc/commit/5bec316))
- Set the project.readme to Markdown ([35d6496](https://github.com/pipefunc/pipefunc/commit/35d6496))
- Make sure to build the package ([a37873a](https://github.com/pipefunc/pipefunc/commit/a37873a))
- use pypa/gh-action-pypi-publish ([2775087](https://github.com/pipefunc/pipefunc/commit/2775087))
- Fix .github/workflows/update-environment.yaml ([58c7b6f](https://github.com/pipefunc/pipefunc/commit/58c7b6f))
- Move lite env and remove jupyterlite_config.json ([22bcd49](https://github.com/pipefunc/pipefunc/commit/22bcd49))
- Fix filename in .github/update-environment.py ([1ffd45f](https://github.com/pipefunc/pipefunc/commit/1ffd45f))
- Update environment.yml ([7c0d005](https://github.com/pipefunc/pipefunc/commit/7c0d005))
- No psutil in jupyterlite ([bcca216](https://github.com/pipefunc/pipefunc/commit/bcca216))
- Update environment.yml ([6d72ab5](https://github.com/pipefunc/pipefunc/commit/6d72ab5))
- Install matplotlib-base in jupyterlite env ([8dc665b](https://github.com/pipefunc/pipefunc/commit/8dc665b))
- Add filename to generate_environment_yml ([b7fed22](https://github.com/pipefunc/pipefunc/commit/b7fed22))
- Refactor .github/update-environment.py ([7375be2](https://github.com/pipefunc/pipefunc/commit/7375be2))
- add jupyterlite-xeus-python as pip only dep ([35ce643](https://github.com/pipefunc/pipefunc/commit/35ce643))
- add jupyterlite_config ([216956b](https://github.com/pipefunc/pipefunc/commit/216956b))
- Add kernel as docs dep ([23a7d7f](https://github.com/pipefunc/pipefunc/commit/23a7d7f))
- Use docs/environment-sphinx.yml for docs building and docs/environment.yml for juyterlite ([85b027c](https://github.com/pipefunc/pipefunc/commit/85b027c))
- Fix jupyterlite-sphinx name ([44bcd6b](https://github.com/pipefunc/pipefunc/commit/44bcd6b))
- Add docs/jupyterlite_config.json ([70fe3ae](https://github.com/pipefunc/pipefunc/commit/70fe3ae))
- Update environment.yml ([73819a3](https://github.com/pipefunc/pipefunc/commit/73819a3))
- add jupyterlite_sphinx ([23b3b20](https://github.com/pipefunc/pipefunc/commit/23b3b20))
- Copy notebook to docs/notebooks ([2765b15](https://github.com/pipefunc/pipefunc/commit/2765b15))
- Move __init__ doc-strings to class top ([ec20e2d](https://github.com/pipefunc/pipefunc/commit/ec20e2d))
- Add example to PipelineFunction ([4ff7796](https://github.com/pipefunc/pipefunc/commit/4ff7796))
- Fix example spacing in doc-string ([5af6054](https://github.com/pipefunc/pipefunc/commit/5af6054))
- Small docs settings changes ([55ba4d7](https://github.com/pipefunc/pipefunc/commit/55ba4d7))
- Rephrase in notebook ([c3f4613](https://github.com/pipefunc/pipefunc/commit/c3f4613))
- Rename readthedocs.yml to .readthedocs.yml ([8b9d8fa](https://github.com/pipefunc/pipefunc/commit/8b9d8fa))
- Remove maxdepth ([6c2d4d5](https://github.com/pipefunc/pipefunc/commit/6c2d4d5))
- chore(docs): update TOC ([2e1cce1](https://github.com/pipefunc/pipefunc/commit/2e1cce1))
- Remove design goals ([3fde37e](https://github.com/pipefunc/pipefunc/commit/3fde37e))
- Links in menu names ([550c3a2](https://github.com/pipefunc/pipefunc/commit/550c3a2))
- Add API docs ([94f206d](https://github.com/pipefunc/pipefunc/commit/94f206d))
- chore(docs): update TOC ([eb48cfa](https://github.com/pipefunc/pipefunc/commit/eb48cfa))
- Add Key Features 🚀 ([38f1508](https://github.com/pipefunc/pipefunc/commit/38f1508))
- Use the help() function ([a4a506c](https://github.com/pipefunc/pipefunc/commit/a4a506c))
- Add tutorial to docs ([568f609](https://github.com/pipefunc/pipefunc/commit/568f609))
- Different pip install optional deps ([633ee65](https://github.com/pipefunc/pipefunc/commit/633ee65))
- Add plotting to docs/environment.yml ([f13e5e7](https://github.com/pipefunc/pipefunc/commit/f13e5e7))
- Update environment.yml ([87baa91](https://github.com/pipefunc/pipefunc/commit/87baa91))
- Add pandas and jupytext as docs dependency ([8825351](https://github.com/pipefunc/pipefunc/commit/8825351))
- Add plotting to docs/environment.yml ([79dcf06](https://github.com/pipefunc/pipefunc/commit/79dcf06))
- Add header image ([eecf416](https://github.com/pipefunc/pipefunc/commit/eecf416))
- Change tagline ([8473f0f](https://github.com/pipefunc/pipefunc/commit/8473f0f))
- Pass through filename ([5fb85ab](https://github.com/pipefunc/pipefunc/commit/5fb85ab))
- chore(docs): update TOC ([d005501](https://github.com/pipefunc/pipefunc/commit/d005501))
- Add example.ipynb ([dde07c0](https://github.com/pipefunc/pipefunc/commit/dde07c0))
- Add tests/test_sweep.py ([59cdd99](https://github.com/pipefunc/pipefunc/commit/59cdd99))
- Add tests/test_pipefunc.py ([2258159](https://github.com/pipefunc/pipefunc/commit/2258159))
- Add tests/test_perf.py ([62b8cfc](https://github.com/pipefunc/pipefunc/commit/62b8cfc))
- Add tests/test_cache.py ([586ca6f](https://github.com/pipefunc/pipefunc/commit/586ca6f))
- Add tests/__init__.py ([85dc61f](https://github.com/pipefunc/pipefunc/commit/85dc61f))
- Add pipefunc/_version.py ([ce2d17f](https://github.com/pipefunc/pipefunc/commit/ce2d17f))
- Add pipefunc/_sweep.py ([a1e2fe5](https://github.com/pipefunc/pipefunc/commit/a1e2fe5))
- Add pipefunc/_plotting.py ([e89850c](https://github.com/pipefunc/pipefunc/commit/e89850c))
- Add pipefunc/_pipefunc.py ([6b94d38](https://github.com/pipefunc/pipefunc/commit/6b94d38))
- Add pipefunc/_perf.py ([8128bd7](https://github.com/pipefunc/pipefunc/commit/8128bd7))
- Add pipefunc/_cache.py ([d25c1a6](https://github.com/pipefunc/pipefunc/commit/d25c1a6))
- Add pipefunc/__init__.py ([0d3cd06](https://github.com/pipefunc/pipefunc/commit/0d3cd06))
- Add docs/source/index.md ([34d6fdb](https://github.com/pipefunc/pipefunc/commit/34d6fdb))
- Add docs/source/conf.py ([526ef73](https://github.com/pipefunc/pipefunc/commit/526ef73))
- Add docs/environment.yml ([053f01a](https://github.com/pipefunc/pipefunc/commit/053f01a))
- Add docs/Makefile ([7b68ec1](https://github.com/pipefunc/pipefunc/commit/7b68ec1))
- Add docs/.gitignore ([6664a3c](https://github.com/pipefunc/pipefunc/commit/6664a3c))
- Add environment.yml ([e82cf48](https://github.com/pipefunc/pipefunc/commit/e82cf48))
- Add setup.cfg ([25c5c63](https://github.com/pipefunc/pipefunc/commit/25c5c63))
- Add readthedocs.yml ([737d480](https://github.com/pipefunc/pipefunc/commit/737d480))
- Add pyproject.toml ([61ec5b9](https://github.com/pipefunc/pipefunc/commit/61ec5b9))
- Add README.md ([b25aebe](https://github.com/pipefunc/pipefunc/commit/b25aebe))
- Add MANIFEST.in ([150e5db](https://github.com/pipefunc/pipefunc/commit/150e5db))
- Add LICENSE ([a467d2e](https://github.com/pipefunc/pipefunc/commit/a467d2e))
- Add AUTHORS.md ([f9175ed](https://github.com/pipefunc/pipefunc/commit/f9175ed))
- Add .pre-commit-config.yaml ([882adb1](https://github.com/pipefunc/pipefunc/commit/882adb1))
- Add .gitignore ([129a4dc](https://github.com/pipefunc/pipefunc/commit/129a4dc))
- Add .github/workflows/update-environment.yaml ([a910911](https://github.com/pipefunc/pipefunc/commit/a910911))
- Add .github/workflows/toc.yaml ([81287cc](https://github.com/pipefunc/pipefunc/commit/81287cc))
- Add .github/workflows/release-drafter.yaml ([ca1a8cf](https://github.com/pipefunc/pipefunc/commit/ca1a8cf))
- Add .github/workflows/pythonpublish.yml ([f73479a](https://github.com/pipefunc/pipefunc/commit/f73479a))
- Add .github/workflows/pytest.yml ([e6ac7eb](https://github.com/pipefunc/pipefunc/commit/e6ac7eb))
- Add .github/workflows/codeql.yml ([1fdfdb6](https://github.com/pipefunc/pipefunc/commit/1fdfdb6))
- Add .github/update-environment.py ([74a4c56](https://github.com/pipefunc/pipefunc/commit/74a4c56))
- Add .github/release-drafter.yml ([0a3ce3d](https://github.com/pipefunc/pipefunc/commit/0a3ce3d))
- Add .gitattributes ([39666ce](https://github.com/pipefunc/pipefunc/commit/39666ce))

### 📊 Stats

- `.toml`: +1 lines, -1 lines
