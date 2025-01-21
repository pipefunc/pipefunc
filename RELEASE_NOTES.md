## Version v0.49.6 (2025-01-17)

### ✨ Enhancements

- ENH: Raise an exception if scope was not added to anything (#571)

## Version v0.49.5 (2025-01-17)

### 🐛 Bug Fixes

- BUG: Fix using `Pipeline.arg_combinations` to calculate `root_args` (#570)

### 📚 Documentation

- DOC: Fix admonition in example.ipynb (#569)
- DOC: Rename `uvtip` -> `try-notebook` and use in `example.ipynb` (#568)
- DOC: Use triple backticks around `uv` command (#567)
- DOC: Add custom `uvtip` directive (#566)
- DOC: Small fixes (#555)

## Version v0.49.4 (2025-01-15)

### 🐛 Bug Fixes

- BUG: Fix `bound` in `NestedPipeFunc` with `scope` and `map` (#560)

### 📚 Documentation

- DOC: Recommendations of order (#559)

## Version v0.49.3 (2025-01-15)

### 🐛 Bug Fixes

- BUG: Fix `bound` in `NestedPipeFunc` inside `Pipeline` (#557)

## Version v0.49.2 (2025-01-14)

### Closed Issues

- NestedPipeFunction in graph show wrong datatype (#487)

### 📚 Documentation

- DOC: Fix propagating defaults in `NestedPipeFunc` (#558)
- DOC: Rename "Benchmarking" to "Overhead and Efficiency" (#553)
- DOC: Add `visualize()` to `basic-usage.md` (#552)
- DOC: Add `opennb` to all examples (#551)
- DOC: Separate out examples into pages (#550)
- DOC: Fix simple typo (#549)
- DOC: Mention `uv` and `opennb` early in tutorial (#548)
- DOC: Reoganize the docs into pages (#545)

### ✨ Enhancements

- ENH: Change the order in which keys appear in `pipeline.info` (#554)

## Version v0.49.1 (2025-01-13)

### 🐛 Bug Fixes

- BUG: Fix `NestedPipeFunction` in graph show wrong datatype (#546)

### 📚 Documentation

- DOC: Add a page about `mapspec` (#543)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#544)

## Version v0.49.0 (2025-01-13)

### ✨ Enhancements

- ENH: Add a widget for `VariantPipeline.visualize()` and `VariantPipeline._repr_mimebundle_` (#539)

### 📚 Documentation

- DOC: Add `NestedPipeFunc` section to example notebooks and move `simplified_pipeline` to FAQ (#542)
- DOC: Fix method name of `Pipeline.join` in example notebook (#541)

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.18 (#538)

## Version v0.48.2 (2025-01-11)

### 🐛 Bug Fixes

- BUG: Add more `NestedPipeFunc` tests and fix multiple outputs issue with them (#536)

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.17 (#535)

### 🧪 Testing

- TST: Add multiple outputs to benchmarks (#537)

## Version v0.48.1 (2025-01-10)

### Closed Issues

- Add pipeline variants (#517)

### 🐛 Bug Fixes

- BUG: Fix scope for `NestedPipeFunc` (#534)

### 🧹 Maintenance

- MAINT: Extend `.gitignore` (#533)

## Version v0.48.0 (2025-01-10)

### Closed Issues

- allow setting names of `NestedPipeFunc` by hand (#195)

### ✨ Enhancements

- ENH: Add `VariantPipelines.from_pipelines` classmethod (#526)
- ENH: Allow setting `NestedPipeFunc(..., function_name="customname")` (#532)

## Version v0.47.3 (2025-01-10)

### 🐛 Bug Fixes

- BUG: Fix `combine_mapspecs` in `NestedPipeFunc` (#531)

## Version v0.47.2 (2025-01-10)

### 🐛 Bug Fixes

- BUG: Set `internal_shape` for `NestedPipeFunc` (#530)
- BUG: Fix error message about using `map_async` with Slurm (#528)
- BUG: Fix case where bound and default are set for same parameter (#525)

### 🤖 CI

- CI: Set `timeout-minutes: 10` in pytest jobs to prevent stuck 6 hour jobs (#529)

### 📚 Documentation

- DOC: Fix FAQ `VariantPipeline` example (#524)

## Version v0.47.1 (2025-01-09)

### 📚 Documentation

- DOC: Add example with non-unique variant names across `PipeFunc`s (#520)

### 🧹 Maintenance

- MAINT: Pin `zarr>=2,<3` (#521)

## Version v0.47.0 (2025-01-09)

### Closed Issues

- Aggregating function outputs into a `dict`? (#456)

### ✨ Enhancements

- ENH: Add `VariantPipeline` that can generate multiple `Pipeline` variants (#518)
- ENH: Add auto-chunksize heuristic (#505)

### 📝 Other

- Use Python 3.13 in CI where possible (#519)
- Update Discord invite link README.md (#509)

### 📦 Dependencies

- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.16 (#516)
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.15 (#514)
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.14 (#511)
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.13 (#508)
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.12 (#507)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#512)
- [pre-commit.ci] pre-commit autoupdate (#504)

### 📚 Documentation

- DOC: Update number of required dependencies in README.md (#506)
- DOC: Autoformat Markdown in FAQ and README (#503)
- DOC: Add comparison with Snakemake (#501)

### 🤖 CI

- CI: Revert `pull_request_target:` (#500)

## Version v0.46.0 (2024-12-23)

### Closed Issues

- Proposal: Reduce Pipeline.map's IPC overhead with chunking (#484)

### 🧪 Testing

- TST: Explicitly set reason in `skipif` (#499)
- TST: Skip shared memory test in CI on nogil Python (3.13t) (#498)

### ✨ Enhancements

- ENH: Allow providing an int to `chunksizes` (#497)
- ENH: Add `chunksizes` argument to `Pipeline.map` and `Pipeline.map_async` (#493)

### 🤖 CI

- CI: Revert `pull_request_target:` for CodSpeed (#495)
- CI: Use `pull_request_target:` to trigger CI on fork (#494)

### 📚 Documentation

- DOC: Mention HPC vs cloud based running (#492)
- DOC: How is this different from Dask, AiiDA, Luigi, Prefect, Kedro, Apache Airflow, etc.? (#491)
- DOC: Add Discord shield (#490)

## Version v0.45.0 (2024-12-21)

### Closed Issues

- Add helpers.getattr (#480)

### ✨ Enhancements

- ENH: Add `size_per_learner` for `SlurmExecutor` (#486)
- ENH: Add `helpers.get_attribute_factory` (#481)

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v5 (#483)
- ⬆️ Update ghcr.io/astral-sh/uv Docker tag to v0.5.11 (#482)

### 📚 Documentation

- DOC: Set Python version to 3.13 in README `opennb` example (#479)
- DOC: Fix header level of "Dynamic Output Shapes and `internal_shapes`" (#478)
- DOC: Small formatting fix in example in doc-string (#477)

## Version v0.44.0 (2024-12-19)

### ✨ Enhancements

- ENH: Add `Pipeline._repr_mimebundle_` (#476)
- ENH: Allow printing rich-formatted table with `pipeline.info()` (#475)
- ENH: Automatically set `internal_shape=("?", ...)` (#463)
- ENH: Add a `.devcontainer` for VS Code based on `uv` (#473)

### 📚 Documentation

- DOC: Update documentation about dynamic `internal_shapes` (#474)

## Version v0.43.0 (2024-12-19)

### Closed Issues

- Allow `internal_shapes` to be input names (str) with simple expressions (#197)

### ✨ Enhancements

- ENH: Enable `show_progress` when using dynamic shapes (#471)
- ENH: Automatically set `internal_shape` (#448)

### 📚 Documentation

- DOC: Add workaround for multiple returns with different sizes (#470)
- DOC: Add `opennb` tip (#464)

### 🐛 Bug Fixes

- BUG: Fix case where there is no size (#467)
- BUG: Ensure to resolve shapes for all arrays in `_update_array` and fix `internal_shape` calculation (#469)
- BUG: Fix case where first dim is "?" (#466)
- BUG: Fix autogenerated `mapspec` issue with mismatching dims check (#465)

## Version v0.42.1 (2024-12-17)

### 🧪 Testing

- TST: Use `pytest-timeout` plugin to prevent handing tests (#459)

### ✨ Enhancements

- ENH: Add `Pipeline.info()` that returns input and output info (#462)

## Version v0.42.0 (2024-12-16)

### ✨ Enhancements

- ENH: Add `pipefunc.helpers.collect_kwargs` helper function (#457)
- ENH: Allow `pipeline.root_args(None)` (default) that returns all inputs (#461)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#460)

## Version v0.41.3 (2024-12-13)

### 🐛 Bug Fixes

- BUG: Only use the cache when `cache=True` instead of always in `pipeline.map` (#458)

## Version v0.41.2 (2024-12-11)

### 🐛 Bug Fixes

- BUG: Fix `internal_shapes` coming from `PipeFunc` constructor and `cleanup=False` (#455)

## Version v0.41.1 (2024-12-11)

### Closed Issues

- Callback on each transition of a good way to visualize the result of each step beyond text (#393)
- Allow per `PipeFunc` storage (#320)
- Allow per `PipeFunc` executor (to mix parallel and local) (#319)
- `TypeError: 'NoneType' object cannot be interpreted as an integer` in documentation build (#317)
- ascii art (#307)

### 🐛 Bug Fixes

- BUG: Fix case with multiple output then iterate over single axis (#454)

### 🧹 Maintenance

- MAINT: Small formatting changes and tiny refactors (from #448) (#453)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#452)
- [pre-commit.ci] pre-commit autoupdate (#445)

### 📝 Other

- Add .ruff_cache to .gitignore (#449)

## Version v0.41.0 (2024-11-27)

### ✨ Enhancements

- ENH: Add `post_execution_hook` for `PipeFunc` (#306)

### 📚 Documentation

- DOC: Set default plotting backend in docs to graphviz (#441)

## Version v0.40.2 (2024-11-27)

### 🧹 Maintenance

- MAINT: Add `pipefunc[all]` to docs extras and remove `pydantic` from `[all]` (#440)
- MAINT: Fix typo (#439)

## Version v0.40.1 (2024-11-27)

### ✨ Enhancements

- ENH: Use `hatch` instead of `setuptools` (#438)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#429)

### 📚 Documentation

- DOC: Fix admonition in example notebook (#437)

## Version v0.40.0 (2024-11-26)

### Closed Issues

- IndexError when handling exceptions without arguments in python<=3.11 (#430)

### 📚 Documentation

- DOC: Add a dropdown with interactive widget explanation (#436)

### ✨ Enhancements

- ENH: Add interactive version of `visualize_graphviz` (#326)
- ENH: Remove pygraphviz dependency, was only used in matplotlib plotting backend (#433)

### 🧹 Maintenance

- MAINT: Sort the dependencies alphabetically (#435)

### 🤖 CI

- CI: Test with plotting in uv now that `pygraphviz` is no longer required (#434)

### 🧪 Testing

- TST: Fix pygraphviz <-> python-graphviz mixup in tests (#432)

## Version v0.39.0 (2024-11-26)

### 🐛 Bug Fixes

- BUG: Fix Python≤3.11 case for `handle_error` (#431)

### 🧹 Maintenance

- MAINT: Install myst-nb with conda (#428)
- MAINT: Remove `LazySequenceLearner` because of alternative in #381 (#419)

### ✨ Enhancements

- ENH: Avoid duplicate dependencies in .github/update-environment.py script (#427)
- ENH: Add support for `pydantic.BaseModel` (#420)
- ENH: Allow using memory based storages in parallel too (#416)

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v4 (#426)
- ⬆️ Update codecov/codecov-action action to v5 (#424)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#425)
- [pre-commit.ci] pre-commit autoupdate (#418)

### 📚 Documentation

- DOC: Fix zarr API docs page (#422)
- DOC: Add section with `dataclass` and `pydantic.BaseModel` (#421)
- DOC: Add ultra-fast bullet point (#417)

## Version v0.38.0 (2024-11-07)

### Closed Issues

- Dataclasses that use default_factory fields have buggy execution on second run (#402)
- Pipeline.add is not idempotent (#394)

### ✨ Enhancements

- ENH: Factor out `SlurmExecutor` logic from `_run.py` (#415)
- ENH: Rename _submit_single to _execute_single to avoid confusion with ex.submit (#413)
- ENH: Allow non-parallel progress bar (#412)
- ENH: Allow using `adaptive_scheduler.SlurmExecutor` (#395)
- ENH: Make `executor` a dict internally always (#410)
- ENH: Prevent duplicates from `PipeFunc`s that return multiple (#409)
- ENH: Add a `StoreType` (#408)
- ENH: Prevent adding functions with same `output_name` (#404)

### 📝 Other

- FIX: Also update progress bar for single executions (#414)
- Define `ShapeDict`, `ShapeTuple`, `UserShapeDict` types (#406)

### 🧪 Testing

- TST: Omit `pipefunc/map/_types.py` from coverage (#411)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#405)

### 🧹 Maintenance

- MAINT: Move `LazySequenceLearner` to separate module (#407)

## Version v0.37.0 (2024-10-30)

### Closed Issues

- All values reported in profile_stats are 0 (#392)

### ✨ Enhancements

- ENH: Specially treat dataclasses with a default factory (closes #402) (#403)
- ENH: Update progress bar every second for first 30 seconds (#401)
- ENH: Include class name in `PipeFunc.__name__` (#389)
- ENH: Add `LazySequenceLearner` (#385)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#400)
- [pre-commit.ci] pre-commit autoupdate (#386)

### 🧹 Maintenance

- MAINT: Split up `_pipeline.py` into modules (#399)
- MAINT: Use relative imports in `pipefunc.map` (#398)
- MAINT: pipefunc.map module reorganization (#397)
- MAINT: Move storage related modules to `map/_storage` (#396)

### 📚 Documentation

- DOC: Fix url in shield (#391)

### 🤖 CI

- CI: Rename GitHub Actions workflows and test with minimal dependencies (#390)

### 📝 Other

- Add `uv` based GitHub Actions workflow and test on free-threaded Python 3.13t (#387)

### 🧪 Testing

- TST: Make optional deps also optional in tests (#388)

## Version v0.36.1 (2024-10-17)

### 🧹 Maintenance

- MAINT: Enable Python 3.13 in CI (#384)

### 📝 Other

- FIX: Use `internal_shapes` defined in `@pipefunc` in `create_learners` (#383)

## Version v0.36.0 (2024-10-16)

### 📝 Other

- Python 3.13 support (#382)

### 📚 Documentation

- DOC: Simplify example in README.md (#379)
- DOC: Add `html_theme_options` (#371)
- DOC: More improvements (#370)
- DOC: Reorder and reorganize docs (#364)
- DOC: Add `sphinx-notfound-page` for 404 (#369)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#377)

### 📦 Dependencies

- ⬆️ Update mamba-org/setup-micromamba action to v2 (#376)

### 🧹 Maintenance

- MAINT: Move `ProgressTracker` widget a `_widgets` folder (#373)

## Version v0.35.1 (2024-09-30)

### ✨ Enhancements

- ENH: Allow pickling `DiskCache` without LRU Cache (#368)
- ENH: Allow `range(...)` as input in `map` (#365)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#366)

### 📚 Documentation

- DOC: Use Ruff badge instead of Black (#367)
- DOC: Improve intro in README (#363)
- DOC: New title and tag line (#362)

## Version v0.35.0 (2024-09-27)

### 📚 Documentation

- DOC: Inline `mapspec` in physics based example (#361)
- DOC: Rely on latest release of MyST (#360)
- DOC: Add FAQ entry about mixing executors and storages (#359)
- DOC: Fix list formatting in Sphinx docs (#358)

### ✨ Enhancements

- ENH: Allow a different `Executor` per `PipeFunc` (#357)
- ENH: Allow setting a `storage` per `PipeFunc` (#356)
- ENH: Fallback to serialization for cache keys (#355)
- ENH: Set `fallback_to_str` to False by default for caching (#354)

## Version v0.34.0 (2024-09-25)

### ✨ Enhancements

- ENH: Add more space between `:` and name in `visualize_graphviz` (#353)
- ENH: Add `pipefunc.testing.patch` (#352)
- ENH: Include mapspec axis in the outputs of `PipeFunc` directly (#349)
- ENH: Keep mapspec in argument nodes in `visualize_graphviz` (#348)

### 📚 Documentation

- DOC: Add mapspec plots to tutorial (#351)

### 🧹 Maintenance

- MAINT: Remove trailing commas to have arg lists on single line (#350)

## Version v0.33.0 (2024-09-24)

### ✨ Enhancements

- ENH: Add `pipeline.map_async` and a progress bar (#333)
- ENH: Raise an error with a helpful error message for missing dependencies (#347)
- ENH: Add optimized `FileArray.mask_linear` (#346)
- ENH: Refactor `pipeline.map.run` to prepare for async implementation (#334)
- ENH: Speedup code by 40% via simple change (#337)
- ENH: Improve missing plotting backend error message (#332)

### 📦 Dependencies

- ⬆️ Update astral-sh/setup-uv action to v3 (#344)

### 🤖 CI

- CI: Remove unused steps from pytest pipeline (#345)

### 🧪 Testing

- TST: Add a CI pipeline that checks for matching doc-strings (#343)
- TST: Add benchmark from FAQ to test suite (#338)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#340)

### 📝 Other

- FIX: Load custom objects correctly in `xarray` (#336)

### 📚 Documentation

- DOC: Add FAQ question about overhead/performance (#335)

## Version v0.32.1 (2024-09-18)

### 📝 Other

- FIX: Improve the parallel store compatibility checking function (#331)

## Version v0.32.0 (2024-09-18)

### Closed Issues

- Add `pipefunc.map.Result.to_xarray` (#312)

### ✨ Enhancements

- ENH: Allow `pipeline.map` to run without disk (#327)
- ENH: Make Graphviz PipeFunc nodes rounded (#329)
- ENH: Implement `graphviz` based visualization (#323)
- ENH: Allow `visualize` to take an int for `figsize` (square) (#322)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#324)

### 📚 Documentation

- DOC: Explain what a pipeline is (#321)
- DOC: Use `DiskCache` to prevent #317 (#318)

## Version v0.31.1 (2024-09-11)

### 📚 Documentation

- DOC: Add a FAQ question about `ErrorSnapshot` and improve IP getting (#316)

### 📝 Other

- Note (#315)

## Version v0.31.0 (2024-09-10)

### ✨ Enhancements

- ENH: Add function going from `Results` to xarray with `xarray_dataset_from_results` (#314)
- ENH: Attach `ErrorSnapshot` for debugging (#313)
- ENH: Use pickle for cache key, inspired by `python-diskcache` package (#310)

### 📚 Documentation

- DOC: Add additional examples to the tutorial (#311)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#308)

### 📝 Other

- Use `repr` for filename key (#309)
- TYP: Fix annotation of `output_picker` (#303)

## Version v0.30.0 (2024-09-05)

### ✨ Enhancements

- ENH: Add `internal_shape` to `PipeFunc` (#302)

### 📚 Documentation

- DOC: Show triangulation on top of `Learner2D` plot (#301)

## Version v0.29.0 (2024-09-05)

### Closed Issues

- Do type validation in pipeline definition (#266)
- Allow caching for `map` (#264)
- allow to inspect the resources inside the function (#192)
- allow internal parallelization (#191)

### ✨ Enhancements

- ENH: Add call to action (#300)
- ENH: Add ToC of questions to FAQ (#298)
- ENH: Add tl;dr note in API docs (#297)
- ENH: Skip parallelization if pointless (#293)
- ENH: Simpler example with `output_picker` (#287)

### 📝 Other

- FIX: Formatting in `is_object_array_type` doc-string (#296)
- FIX: formatting of lists in doc-strings (#295)
- FIX: doc-string of `func_dependents` and `func_dependencies` (#294)
- FIX: Correctly set cache value for `HybridCache` (#292)
- Allow to use cache for `Pipeline.map` (#291)
- Add `pipefunc.cache` and `pipefunc.typing` to the reference documentation (#290)
- Add `.cache` attribute to function using `@memoize` (#288)

## Version v0.28.0 (2024-09-03)

### 📝 Other

- Rename `pipefunc._cache` to `pipefunc.cache` (#286)
- Update `asciinema` recording (#281)
- Add asciinema recording (#280)
- Build `dirhtml` Sphinx docs instead of `html` (#279)

### ✨ Enhancements

- ENH: Small type annotation fix in `memoize` (#285)
- ENH: Improve caching and add a `memoize` decorator (#283)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#284)

## Version v0.27.3 (2024-08-29)

### 📝 Other

- FIX: Case where reduction happens and output is unresolvable (#278)
- Add `py.typed` (PEP 561) (#277)

## Version v0.27.2 (2024-08-29)

### 📝 Other

- Fix type annotation bug with autogenerated axis with internal shape (#276)

## Version v0.27.1 (2024-08-29)

### 📝 Other

- Skip on `NoAnnotation` (#275)
- Add type annotation checking documentation (#274)
- Enforce one-to-one mapping for renames and improve validation error messages (#273)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#267)

## Version v0.27.0 (2024-08-28)

### 📝 Other

- Allow disabling type validation (#271)
- Allow types to be generics (#269)
- Ignore ARG001 ruff rule in tests (#270)
- Try getting type-hints instead of allowing to error out (#268)
- Add parameter and output annotations and validate them during `Pipeline` construction (#6)
- Simplify Adaptive Scheduler code (#263)
- Set Ruff Python version to 3.10 (#262)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#259)

## Version v0.26.0 (2024-08-22)

### 📝 Other

- Allow single job per element inside a `MapSpec` via `resources_scope` (#260)
- Return correct data in SequenceLearner when `return_output` (#261)
- Add `pipeline.run` adaptive tools (#257)
- Remove indentation level (#255)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#254)
- [pre-commit.ci] pre-commit autoupdate (#253)
- [pre-commit.ci] pre-commit autoupdate (#252)
- [pre-commit.ci] pre-commit autoupdate (#250)

### 📦 Dependencies

- ⬆️ Update CodSpeedHQ/action action to v3 (#251)

## Version v0.25.0 (2024-07-19)

### 📝 Other

- Add `parallelization_mode` option (#249)

## Version v0.24.0 (2024-07-18)

### Closed Issues

- AssertionError raised in the case of a function without inputs. (#238)

### 📝 Other

- Make Resources serializable (#247)
- Support delayed `Resources` in Adaptive Scheduler integration (#234)
- Rename `Resources` attributes `cpus`, `gpus`, `nodes`, `cpus_per_node`, `time` (#245)
- Split parts of `test_pipefunc.py` into several files (#242)
- Raise an exception when parameters and output_name overlaps (#241)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#246)
- [pre-commit.ci] pre-commit autoupdate (#244)

## Version v0.23.1 (2024-06-28)

### 📝 Other

- Allow parameterless functions in a Pipeline (#240)
- Allow passing `loss_function` to `to_adaptive_learner` (#239)

## Version v0.23.0 (2024-06-27)

### 📝 Other

- Add a poor man's adaptive integration (#237)

## Version v0.22.2 (2024-06-27)

### 📝 Other

- Disallow mapping over bound arguments and fix mapping over defaults (#236)

## Version v0.22.1 (2024-06-27)

### 📝 Other

- Always call validate in `add` to ensure mapspec axes are autogenerated (#235)

## Version v0.22.0 (2024-06-26)

### 📝 Other

- Get rid of `PipeFunc._default_resources` (#232)
- Allow bound arguments to be unhashable (#233)
- Allow `resources` to be delayed via a `Callable[[dict], Resources]` (#219)
- Add `resources_variable` in `PipeFunc`  (#220)
- Fix the Python version for Codspeed (#231)

## Version v0.21.0 (2024-06-24)

### Closed Issues

- Changing PipeFunc should trigger Pipeline internal cache reset (#203)

### 📝 Other

- Fix `dev` section in pyproject.toml `[project.optional-dependencies]`
- Fix `PipeFunc` that share defaults (#230)
- Add Codspeed speedtest/benchmarking CI (#229)
- Add Renovate CI integration (#221)
- Combine resources with default_resources in `PipeFunc` object (#214)
- Simplify `Pipeline.copy` (#217)
- Remove `PipeFunc.__getattr__` and define `PipeFunc.__name__` (#216)
- Always create a copy when calling `Pipeline.add` (#215)
- Keep `default_resources` in `Pipeline` and rename `resources_report` to `print_profiling_stats` to avoid confusion (#213)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#228)

### 📦 Dependencies

- ⬆️ Update release-drafter/release-drafter action to v6 (#227)
- ⬆️ Update github/codeql-action action to v3 (#226)
- ⬆️ Update actions/setup-python action to v5 (#225)
- ⬆️ Update release-drafter/release-drafter action to v5.25.0 (#222)
- ⬆️ Update actions/checkout action to v4 (#223)

## Version v0.20.0 (2024-06-19)

### 📝 Other

- Remove specialized Adaptive code, and generalize `map` (#212)
- Remove `save_function` from `PipeFunc` and `delayed_callback` from `_LazyFunction` (#211)
- Remove the PipeFunc.set_profile method (#210)
- Factor out `_MockPipeline` (#209)
- Use frozen and slotted dataclasses where possible (#208)
- Keep a WeakRef to the Pipeline in each PipeFunc to reset Pipeline cache (#207)

## Version v0.19.0 (2024-06-17)

### 📝 Other

- Introduce parameter namespacing via `scope`s (#201)
- Make sure all cells are executed to ensure working docs (#206)
- Create a copy of a `PipeFunc` in `Pipeline.add` (#205)

## Version v0.18.1 (2024-06-14)

### Closed Issues

- Rename outputs too in `update_renames` (#189)

### 📝 Other

- Clear internal cache after renaming and re-defaulting (#202)

## Version v0.18.0 (2024-06-13)

### Closed Issues

- include single results in xarray (#188)
- Rename mapspecs in update_remames (#184)

### 📝 Other

- Allow renaming `output_name` in `update_renames` (#200)
- Set `run_folder=None` by default (#198)
- Rename `MapSpec` in `update_renames` (#196)
- Add FAQ (#187)
- Include single results as 0D arrays in `xarray.Dataset` (#190)
- Extend `to_slurm_run` to return `adaptive_scheduler.RunManager` (#186)
- Add edge to `NestedPipeFunc` (#183)
- Update `example.ipynb` tutorial (#182)

## Version v0.17.0 (2024-06-11)

### 📝 Other

- Add remaining `Sweep` tests to reach 100% coverage on all code :tada: (#181)
- Remove superseded sweep functions: `get_precalculation_order` and `get_min_sweep_sets` (#180)
- Allow passing `update_from` to `update_renames` (#179)
- Fix regression introduced in #156 (#178)
- Reimplement `Pipeline.simplified_pipeline` using `NestedPipeFunc` (#156)
- Reach 100% testing coverage in `pipefunc/_pipeline.py` (#177)
- Increase testing coverage (#176)
- Fix typo and add more references

## Version v0.16.0 (2024-06-10)

### 📝 Other

- Add pipeline.update_rename to example
- Add `Pipeline.update_renames` (#175)
- Allow to nest all (#174)
- Add `Pipeline.nest_funcs` (#173)
- Do not rely on hashing when checking defaults (#172)
- Deal with unhashable defaults (#171)
- Add sanity checks (#170)
- Add `Pipeline.update_defaults` (#169)
- Add `pipeline.join` and `pipeline1 | pipeline2` (#168)
- HoloViews plotting improvements (#166)

## Version v0.15.1 (2024-06-07)

### 📝 Other

- Do not add `MapSpec` axis for bound parameters (#165)

## Version v0.15.0 (2024-06-07)

### Closed Issues

- class CombinedFunc(PipeFunc) to nest pipelines (#138)

### 📝 Other

- Make bound values actual node types in the graph (#160)
- Fix setting `__version__` during onbuild (#164)
- Pass through `internal_shapes` in `create_learners` (#162)
- Use `xarray.merge(... compat="override")` to deal with merging issues (#161)
- Add missing API docs file for `pipefunc.map.adaptive_scheduler`
- Add Adaptive Scheduler integration (#159)
- Mention Xarray earlier in the docs
- Make `resources` a module (#158)
- Disallow spaces in `Resources(memory)`
- Implement `resources` specification (#157)

## Version v0.14.0 (2024-06-04)

### 📝 Other

- Reorder functions, put public code at top of modules (#155)
- Add `NestedPipeFunc` (#153)
- Set author in documentation to PipeFunc Developers
- Fix typo
- Rename to `auto_subpipeline` (#150)
- Add option to pick the `output_name` and partial inputs when running `pipeline.map` (#127)
- Include `pipefunc.map.adaptive` integration in docs (#149)
- Validate inputs to `PipeFunc` (#148)
- Add `PipeFunc.update_bound` to allow fixed parameters (#110)
- Make `versioningit` an optional runtime dependency (#144)
- Set MyST in .github/update-environment.py (#143)

## Version v0.13.0 (2024-06-02)

### 📝 Other

- Fix `pipeline.mapspecs_as_strings` statement (which is a property now)
- Drop support for Python 3.8 and 3.9 (#142)
- Factor out simplify functions to simplify module (#141)
- Factor out `resources_report` (#140)
- Make more `cached_property`s (#139)
- Make `PipeFunc.renames` a property to avoid mutation (#137)
- Copy defaults in copy method (#135)
- Define many independent Adaptive learners for cross-products (#136)
- Implement `pipeline.map(... fixed_indices)` which computes the output only for selected indices (#129)
- Add `PipeFunc.update_renames` and `PipeFunc.update_defaults` (#128)
- Remove unused helper functions to join sets and find common items (#134)
- Make `pipefunc.lazy` a public module (#133)
- Cleanup `__init__.py` and make the `sweep` module public (#132)
- Use `sphinx-autodoc-typehints` (#131)
- Rename `map_parameters` to `mapspec_names` (#130)
- Validate inputs when calling `pipeline.map` (#126)
- Documentation MyST fixes and style changes (#125)
- Parallel docs changes (#124)
- Parallelize all functions in the same generation (#123)
- Factor out `RunInfo` to separate module (#122)
- Add `Pipeline.replace` (#121)
- add `join_overlapping_sets` and `common_in_sets` (#120)
- Allow setting new defaults in `PipeFunc` (#111)

## Version v0.12.0 (2024-05-30)

### 📝 Other

- Add custom parallelism section to the docs (#119)
- Add `SharedDictArray` (#118)
- Revert _SharedDictStore name change
- Fix typo in test function name
- Implement native `DictArray` (#117)
- Transfer to `github.com/pipefunc` org (#116)
- Add tests for `Pipeline.independent_axes_in_mapspecs` (#115)
- Functionality to identify independent axes in collection of `MapSpec`s (#84)
- Store `RunInfo` as JSON instead of cloudpickled bytes (#109)
- Allow passing any `concurrent.futures.Executor` (#108)
- Rename `ZarrArray` to `ZarrFileArray` (#107)
- Add mention of Xarray and Zarr
- Add `ZarrMemory` and `ZarrSharedMemory` (#106)
- Fix headers in API docs

## Version v0.11.0 (2024-05-28)

### 📝 Other

- Pass through storage and return `map` result as `dict[str, Result]` (#104)
- Test all storage and remove custom `gzip` cloudpickle because `zarr` already compresses (#103)
- Add `zarr` integration (#101)
- Fix `map` run order (#102)
- Add `MapSpec` in `Pipeline.visualize` (#72)
- Mention where example is based on

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#105)

## Version v0.10.0 (2024-05-24)

### 📝 Other

- Add `xarray` integration (#94)
- Make sure to only evaluate a function once when possible (#100)
- Only create cache if functions have caching enabled (#99)
- Use sphinx-book-theme instead of furo (#98)
- Make `output_to_func` a `cached_property` and `RunInfo` a `dataclass`, and some renames (#96)
- Replace `tabulate` dependency by simple function (#97)

## Version v0.9.0 (2024-05-22)

### 📝 Other

- Add support for output arrays with internal structure and autogenerate MapSpecs (#85)
- Style changes (#93)
- Allow calling `add_mapspec_axis` on multiple parameters (#92)
- Rename `manual_shapes` to `internal_shapes` (#91)
- Fix bug and refactor `FileArray` (#90)
- Add `PipeFunc.copy()` and use it when creating `Pipeline` with tuples including `MapSpec`s (#89)
- Implement `FileArray` with internal structure (#88)
- `MapSpec` method changes and add `Pipeline.mapspec_axes` and `mapspec_dimensions` (#86)
- Rephrase doc-string
- Add zipping axis test and doc-string (#83)
- Create a temporary `run_folder` if `None` and `README.md` improvements (#82)
- Remove fan-out/fan-in
- Add `mapspecs` method, `sorted_functions` property, and rewrite intro in `README` (#81)
- Better error message in `Pipeline.run` (#80)
- Fix bug for `add_mapspec_axis` (#79)
- Add `Pipeline.add_mapspec_axis` for cross-products (#78)
- Create separate API docs per module (#77)
- Fix header in example.ipynb
- Reorder the docs and small rewrite (#76)
- Add docs section about renames (#75)
- Dump to `FileArray` as soon as possible (#74)
- Fix typo in docs and cache improvements (#73)

## Version v0.8.0 (2024-05-17)

### 📝 Other

- Increase coverage and fix Sweep bug (#71)
- Add verbose flag (#70)
- Remove `_update_wrapper` to make `dataclass`es pickleble (#69)
- Compare `RunInfo` to old saved `RunInfo` (#68)
- Add picklable `_MapWrapper` used in `create_learners_from_sweep` (#67)
- Add loading of data that already exists in `Pipeline.map` (#66)
- Rename `get_cache` to `_current_cache` (#63)
- Rename to `_run_pipeline` to `run` to align with `map` (#64)

## Version v0.7.0 (2024-05-15)

### 📝 Other

- Add pipefunc.map.adaptive to API docs
- Better `resource_report` and add add `Sweep` with `MapSpec` tools (#62)
- Add `pipefunc.map` to API docs
- Use updated logo
- Docs improvements (#61)
- Remove Jupyterlite configuration
- Add Map-Reduce to features list
- Add `Pipeline.map` docs and automatically parallelize `map` (#59)
- Various small improvements (#58)
- Style changes (100 character lines) (#57)

## Version v0.6.0 (2024-05-15)

### 📝 Other

- Integrate `MapSpec`ed `Pipeline`s with Adaptive (#56)
- Add functionality to run `Pipeline`s with `MapSpec`s (#55)
- Refactor, improve, test, and integrate `MapSpec` into `Pipeline` (#22)
- Add `MapSpec` and `FileBasedObjectArray` `from aiida-dynamic-workflows` (#54)
- Improve utils, add topological_generations, and better error message (#53)
- Fix docs (jupyterlite) (#52)
- Take out arg_combination functions (#51)
- Take out methods and make functions and simplify code (#50)
- dump, load, Pipeline.defaults, Pipeline.copy, and style (#49)
- `construct_dag` fix and remove dead code (#47)
- Refactor `Pipeline._execute_pipeline` (#44)
- Switch around log message (#45)
- Add test `test_full_output_cache` (#46)
- Fix test_handle_error on MacOS (#43)
- Better error message (#42)
- Raise when unused parameters are provided (#41)
- Add pipeline.drop (#40)
- Rename PipelineFunction -> PipeFunc (#39)
- Several caching fixes (#38)
- Use codecov/codecov-action@v4 (#36)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#48)
- [pre-commit.ci] pre-commit autoupdate (#37)

## Version v0.5.0 (2024-04-30)

### 📝 Other

- Make positional only (#35)
- Format line
- Remove unused var T
- Reorganize some definitions into modules (#34)
- Add a TaskGraph (#33)
- Fix cache argument in docs and fix pickling issues (#32)
- Add 3.12 to testing matrix (#31)
- Optimizations (#30)
- Rename cloudpickle parameter (#29)
- Allow lazy pipeline evaluation (#26)
- Add Cache ABC (#28)
- Cache improvement and rename (#27)
- Add `with_cloudpickle` to `HybridCache` (#25)
- Add `DiskCache` (#24)
- Add root_args method (#23)
- Add hype tag (AI)
- Rewrite the intro in the README (#21)

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#20)
- [pre-commit.ci] pre-commit autoupdate (#19)
- [pre-commit.ci] pre-commit autoupdate (#18)
- [pre-commit.ci] pre-commit autoupdate (#17)

## Version v0.4.0 (2024-03-11)

### 📝 Other

- Keep functions picklable
- Use kwargs with derivers
- Fix typo
- Rename callables to derivers (#16)
- Do not overwrite keys that exist in the sweep (#15)
- Add `callables` and `product` to Sweep (#14)
- Use Pipeline.leaf_nodes instead of unique tip (#12)
- Call `update_wrapper` for correct signature (#11)

## Version v0.3.0 (2024-03-08)

### 📝 Other

- Automatically set `output_name` if possible (#10)
- Unique colors for combinable and non-combinable nodes (#9)
- Fix coloring of combinable nodes (#8)
- Allow constants in `Sweep`s (#7)
- Allow constants in a Sweep (#4)
- Fix line length (#5)
- Color combinable and add test
- remove cell
- Rename reduce -> simplify
- Remove incorrect copyright message
- Update environment.yml
- Skip plotting
- Fix test dependencies (add pandas)
- More pre-commit and typing fixes
- Use ruff-format
- Fix pre-commit issues
- Update pre-commit filters
- Fix pip install command in README.md

### 🔄 Pre-commit

- [pre-commit.ci] pre-commit autoupdate (#3)

## Version v0.2.0 (2023-11-27)

### Closed Issues

- Header (#1)

### 📝 Other

- Add Python 3.12 classifier
- Fix doc-string
- Remove print statement
- Install black[jupyter] in dev deps
- Add saving
- Unshallow clone in Readthedocs
- Add shields
- Fix type hint in _sweep.py
- Another typo fix in all_transitive_paths
- Fix type in doc-string
- Add all_transitive_paths to get parallel and indepent computation chains
- Add leaf and root nodes property
- Fix _assert_valid_sweep_dict
- Add get_min_sweep_sets
- Rewrap text in doc-strings
- Add all_execution_orders
- Add conservatively_combine
- rename 'add' to 'combine'
- Remove [project.scripts] section from pyproject.toml

## Version v0.1.0 (2023-07-16)

### 📝 Other

- Fix license in pyproject.toml
- Set the project.readme to Markdown
- Make sure to build the package
- use pypa/gh-action-pypi-publish
- Fix .github/workflows/update-environment.yaml
- Move lite env and remove jupyterlite_config.json
- Fix filename in .github/update-environment.py
- Update environment.yml
- No psutil in jupyterlite
- Update environment.yml
- Install matplotlib-base in jupyterlite env
- Add filename to generate_environment_yml
- Refactor .github/update-environment.py
- add jupyterlite-xeus-python as pip only dep
- add jupyterlite_config
- Add kernel as docs dep
- Use docs/environment-sphinx.yml for docs building and docs/environment.yml for juyterlite
- Fix jupyterlite-sphinx name
- Add docs/jupyterlite_config.json
- Update environment.yml
- add jupyterlite_sphinx
- Copy notebook to docs/notebooks
- Move __init__ doc-strings to class top
- Add example to PipelineFunction
- Fix example spacing in doc-string
- Small docs settings changes
- Rephrase in notebook
- Rename readthedocs.yml to .readthedocs.yml
- Remove maxdepth
- chore(docs): update TOC
- Remove design goals
- Links in menu names
- Add API docs
- chore(docs): update TOC
- Add Key Features 🚀
- Use the help() function
- Add tutorial to docs
- Different pip install optional deps
- Add plotting to docs/environment.yml
- Update environment.yml
- Add pandas and jupytext as docs dependency
- Add plotting to docs/environment.yml
- Add header image
- Change tagline
- Pass through filename
- chore(docs): update TOC
- Add example.ipynb
- Add tests/test_sweep.py
- Add tests/test_pipefunc.py
- Add tests/test_perf.py
- Add tests/test_cache.py
- Add tests/__init__.py
- Add pipefunc/_version.py
- Add pipefunc/_sweep.py
- Add pipefunc/_plotting.py
- Add pipefunc/_pipefunc.py
- Add pipefunc/_perf.py
- Add pipefunc/_cache.py
- Add pipefunc/__init__.py
- Add docs/source/index.md
- Add docs/source/conf.py
- Add docs/environment.yml
- Add docs/Makefile
- Add docs/.gitignore
- Add environment.yml
- Add setup.cfg
- Add readthedocs.yml
- Add pyproject.toml
- Add README.md
- Add MANIFEST.in
- Add LICENSE
- Add AUTHORS.md
- Add .pre-commit-config.yaml
- Add .gitignore
- Add .github/workflows/update-environment.yaml
- Add .github/workflows/toc.yaml
- Add .github/workflows/release-drafter.yaml
- Add .github/workflows/pythonpublish.yml
- Add .github/workflows/pytest.yml
- Add .github/workflows/codeql.yml
- Add .github/update-environment.py
- Add .github/release-drafter.yml
- Add .gitattributes
