from __future__ import annotations

import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from pipefunc import PipeFunc, Pipeline
from pipefunc._utils import assert_complete_kwargs, is_installed, is_running_in_ipynb, requires

if TYPE_CHECKING:
    from collections.abc import Callable

    import ipywidgets

    from pipefunc._pipefunc import PipeFunc


class VariantPipeline:
    """A pipeline container that supports multiple implementations (variants) of functions.

    `VariantPipeline` allows you to define multiple implementations of functions and
    select which variant to use at runtime. This is particularly useful for:

    - A/B testing different implementations
    - Experimenting with algorithm variations
    - Managing multiple processing options
    - Creating configurable pipelines

    The pipeline can have multiple variant groups, where each group contains alternative
    implementations of a function. Functions can be assigned to variant groups using
    the ``variant`` parameter which can be a single string (for the default group) or
    a dictionary mapping group names to variant names.

    All parameters below (except ``functions`` and ``default_variant``) are simply passed to
    the `~pipefunc.Pipeline` constructor when creating a new pipeline with the selected
    variant(s) using the `with_variant` method.

    Parameters
    ----------
    functions
        List of `PipeFunc` instances.
    default_variant
        Default variant to use if none is specified in `with_variant`.
        Either a single variant name or a dictionary mapping variant
        groups to variants.
    lazy
        Flag indicating whether the pipeline should be lazy.
    debug
        Flag indicating whether debug information should be printed.
        If ``None``, the value of each PipeFunc's debug attribute is used.
    profile
        Flag indicating whether profiling information should be collected.
        If ``None``, the value of each PipeFunc's profile attribute is used.
        Profiling is only available for sequential execution.
    cache_type
        The type of cache to use. See the notes below for more *important* information.
    cache_kwargs
        Keyword arguments passed to the cache constructor.
    validate_type_annotations
        Flag indicating whether type validation should be performed. If ``True``,
        the type annotations of the functions are validated during the pipeline
        initialization. If ``False``, the type annotations are not validated.
    scope
        If provided, *all* parameter names and output names of the pipeline functions will
        be prefixed with the specified scope followed by a dot (``'.'``), e.g., parameter
        ``x`` with scope ``foo`` becomes ``foo.x``. This allows multiple functions in a
        pipeline to have parameters with the same name without conflict. To be selective
        about which parameters and outputs to include in the scope, use the
        `Pipeline.update_scope` method.

        When providing parameter values for pipelines that have scopes, they can
        be provided either as a dictionary for the scope, or by using the
        ``f'{scope}.{name}'`` notation. For example,
        a `Pipeline` instance with scope "foo" and "bar", the parameters
        can be provided as:
        ``pipeline(output_name, foo=dict(a=1, b=2), bar=dict(a=3, b=4))`` or
        ``pipeline(output_name, **{"foo.a": 1, "foo.b": 2, "bar.a": 3, "bar.b": 4})``.
    default_resources
        Default resources to use for the pipeline functions. If ``None``,
        the resources are not set. Either a dict or a `pipefunc.resources.Resources`
        instance can be provided. If provided, the resources in the `PipeFunc`
        instances are updated with the default resources.

    Examples
    --------
    Simple variant selection:

        >>> @pipefunc(output_name="c", variant="add")
        ... def f(a, b):
        ...     return a + b
        ...
        >>> @pipefunc(output_name="c", variant="sub")
        ... def f_alt(a, b):
        ...     return a - b
        ...
        >>> @pipefunc(output_name="d")
        ... def g(b, c):
        ...     return b * c
        ...
        >>> pipeline = VariantPipeline([f, f_alt, g], default_variant="add")
        >>> pipeline_add = pipeline.with_variant()  # Uses default variant
        >>> pipeline_sub = pipeline.with_variant(select="sub")
        >>> pipeline_add(a=2, b=3)  # (2 + 3) * 3 = 15
        15
        >>> pipeline_sub(a=2, b=3)  # (2 - 3) * 3 = -3
        -3

    Multiple variant groups:

        >>> @pipefunc(output_name="c", variant={"method": "add"})
        ... def f1(a, b):
        ...     return a + b
        ...
        >>> @pipefunc(output_name="c", variant={"method": "sub"})
        ... def f2(a, b):
        ...     return a - b
        ...
        >>> @pipefunc(output_name="d", variant={"analysis": "mul"})
        ... def g1(b, c):
        ...     return b * c
        ...
        >>> @pipefunc(output_name="d", variant={"analysis": "div"})
        ... def g2(b, c):
        ...     return b / c
        ...
        >>> pipeline = VariantPipeline(
        ...     [f1, f2, g1, g2],
        ...     default_variant={"method": "add", "analysis": "mul"}
        ... )
        >>> # Select specific variants for each group
        >>> pipeline_sub_div = pipeline.with_variant(
        ...     select={"method": "sub", "analysis": "div"}
        ... )

    Notes
    -----
    - Functions without variants can be included in the pipeline and will be used
      regardless of variant selection.
    - When using ``with_variant()``, if all variants are resolved, a regular `~pipefunc.Pipeline`
      is returned. If some variants remain unselected, another `VariantPipeline` is
      returned.
    - The ``default_variant`` can be a single string (if there's only one variant group)
      or a dictionary mapping variant groups to their default variants.
    - Variants in the same group can have different output names, allowing for
      flexible pipeline structures.

    See Also
    --------
    pipefunc.Pipeline
        The base pipeline class.
    pipefunc.PipeFunc
        Function wrapper that supports variants.

    """

    def __init__(
        self,
        functions: list[PipeFunc],
        *,
        default_variant: str | dict[str | None, str] | None = None,
        lazy: bool = False,
        debug: bool | None = None,
        profile: bool | None = None,
        cache_type: Literal["lru", "hybrid", "disk", "simple"] | None = None,
        cache_kwargs: dict[str, Any] | None = None,
        validate_type_annotations: bool = True,
        scope: str | None = None,
        default_resources: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a VariantPipeline."""
        self.functions = functions
        self.default_variant = default_variant
        self.lazy = lazy
        self.debug = debug
        self.profile = profile
        self.cache_type = cache_type
        self.cache_kwargs = cache_kwargs
        self.validate_type_annotations = validate_type_annotations
        self.scope = scope
        self.default_resources = default_resources
        if not self.variants_mapping():
            msg = "No variants found in the pipeline. Use a regular `Pipeline` instead."
            raise ValueError(msg)

    def variants_mapping(self) -> dict[str | None, set[str]]:
        """Return a dictionary of variant groups and their variants."""
        variant_groups: dict[str | None, set[str]] = {}
        for function in self.functions:
            for group, variant in function.variant.items():
                variants = variant_groups.setdefault(group, set())
                variants.add(variant)
        return variant_groups

    def _variants_mapping_inverse(self) -> dict[str, set[str | None]]:
        """Return a dictionary of variants and their variant groups."""
        variants: dict[str, set[str | None]] = {}
        for function in self.functions:
            for group, variant in function.variant.items():
                groups = variants.setdefault(variant, set())
                groups.add(group)
        return variants

    def with_variant(
        self,
        select: str | dict[str | None, str] | None = None,
        **kwargs: Any,
    ) -> Pipeline | VariantPipeline:
        """Create a new Pipeline or VariantPipeline with the specified variant selected.

        Parameters
        ----------
        select
            Name of the variant to select. If not provided, `default_variant` is used.
            If `select` is a string, it selects a single variant if no ambiguity exists.
            If `select` is a dictionary, it selects a variant for each variant group, where
            the keys are variant group names and the values are variant names.
            If a partial dictionary is provided (not covering all variant groups) and
            default_variant is a dictionary, it will merge the defaults with the selection.
        kwargs
            Keyword arguments for changing the parameters for a Pipeline or VariantPipeline.

        Returns
        -------
            A new Pipeline or VariantPipeline with the selected variant(s).
            If variants remain, a VariantPipeline is returned.
            If no variants remain, a Pipeline is returned.

        Raises
        ------
        ValueError
            If the specified variant is ambiguous or unknown, or if an invalid variant type is provided.
        TypeError
            If `select` is not a string or a dictionary.

        """
        if select is None:
            if self.default_variant is None:
                msg = "No variant selected and no default variant provided."
                raise ValueError(msg)
            select = self.default_variant

        if isinstance(select, str):
            select = self._resolve_single_variant(select)
        elif not isinstance(select, dict):
            msg = f"Invalid variant type: `{type(select)}`. Expected `str` or `dict`."
            raise TypeError(msg)

        if isinstance(self.default_variant, dict):
            select = self.default_variant | select

        assert isinstance(select, dict)
        _validate_variants_exist(self.variants_mapping(), select)

        new_functions = self._select_functions(select)
        variants_remain = self._check_remaining_variants(new_functions)

        if variants_remain:
            return self.copy(functions=new_functions, **kwargs)

        # No variants left, return a regular Pipeline
        return Pipeline(
            new_functions,  # type: ignore[arg-type]
            lazy=kwargs.get("lazy", self.lazy),
            debug=kwargs.get("debug", self.debug),
            profile=kwargs.get("profile", self.profile),
            cache_type=kwargs.get("cache_type", self.cache_type),
            cache_kwargs=kwargs.get("cache_kwargs", self.cache_kwargs),
            validate_type_annotations=kwargs.get(
                "validate_type_annotations",
                self.validate_type_annotations,
            ),
            scope=kwargs.get("scope", self.scope),
            default_resources=kwargs.get("default_resources", self.default_resources),
        )

    def _resolve_single_variant(self, select: str) -> dict[str | None, str]:
        """Resolve a single variant string to a dictionary."""
        inv = self._variants_mapping_inverse()
        group = inv.get(select, set())
        if len(group) > 1:
            msg = f"Ambiguous variant: `{select}`, could be in either `{group}`"
            raise ValueError(msg)
        if not group:
            msg = f"Unknown variant: `{select}`, choose one of: `{', '.join(inv)}`"
            raise ValueError(msg)
        return {group.pop(): select}

    def _select_functions(self, select: dict[str | None, str]) -> list[PipeFunc]:
        """Select functions based on the given variant selection."""
        new_functions: list[PipeFunc] = []
        for function in self.functions:
            # For functions with no variants, always include them
            if not function.variant:
                new_functions.append(function)
                continue

            # Check if function matches the selected variants
            include = True

            # Check variants dict
            for group, variant in function.variant.items():
                if group in select and select[group] != variant:
                    include = False
                    break

            if include:
                new_functions.append(function)

        return new_functions

    def _check_remaining_variants(self, functions: list[PipeFunc]) -> bool:
        """Check if any variants remain after selection."""
        left_over = defaultdict(set)
        for function in functions:
            for group, variant in function.variant.items():
                left_over[group].add(variant)
        return any(len(variants) > 1 for variants in left_over.values())

    def copy(self, **kwargs: Any) -> VariantPipeline:
        """Return a copy of the VariantPipeline.

        Parameters
        ----------
        kwargs
            Keyword arguments passed to the `VariantPipeline` constructor instead of the
            original values.

        """
        original_kwargs = {
            "functions": self.functions,
            "lazy": self.lazy,
            "debug": self.debug,
            "profile": self.profile,
            "cache_type": self.cache_type,
            "cache_kwargs": self.cache_kwargs,
            "validate_type_annotations": self.validate_type_annotations,
            "scope": self.scope,
            "default_resources": self.default_resources,
            "default_variant": self.default_variant,
        }
        assert_complete_kwargs(original_kwargs, VariantPipeline.__init__, skip={"self"})
        original_kwargs.update(kwargs)
        return VariantPipeline(**original_kwargs)  # type: ignore[arg-type]

    @classmethod
    def from_pipelines(
        cls,
        *variant_pipeline: tuple[str, str, Pipeline] | tuple[str, Pipeline],
    ) -> VariantPipeline:
        """Create a new `VariantPipeline` from multiple `Pipeline` instances.

        This method constructs a `VariantPipeline` by combining functions from
        multiple `Pipeline` instances, identifying common functions and assigning
        variants based on the input tuples.

        Each input tuple can either be a 2-tuple or a 3-tuple.
        - A 2-tuple contains: ``(variant_name, pipeline)``.
        - A 3-tuple contains: ``(variant_group, variant_name, pipeline)``.

        Functions that are identical across all input pipelines (as determined by
        the `is_identical_pipefunc` function) are considered "common" and are added to
        the resulting `VariantPipeline` without any variant information.

        Functions that are unique to a specific pipeline are added with their
        corresponding variant information (if provided in the input tuple).

        Parameters
        ----------
        *variant_pipeline
            Variable number of tuples, where each tuple represents a pipeline and its
            associated variant information. Each tuple can be either:
            - `(variant_name, pipeline)`: Specifies the variant name for all functions
            in the pipeline. The variant group will be set to `None` (default group).
            - `(variant_group, variant_name, pipeline)`: Specifies both the variant
            group and variant name for all functions in the pipeline.

        Returns
        -------
            A new `VariantPipeline` instance containing the combined functions from
            the input pipelines, with appropriate variant assignments.

        Examples
        --------
        >>> @pipefunc(output_name="x")
        ... def f(a, b):
        ...     return a + b
        ...
        >>> @pipefunc(output_name="y")
        ... def g(x, c):
        ...     return x * c
        ...
        >>> pipeline1 = Pipeline([f, g])
        >>> pipeline2 = Pipeline([f, g.copy(func=lambda x, c: x / c)])
        >>> variant_pipeline = VariantPipeline.from_pipelines(
        ...     ("add_mul", pipeline1),
        ...     ("add_div", pipeline2)
        ... )
        >>> add_mul_pipeline = variant_pipeline.with_variant(select="add_mul")
        >>> add_div_pipeline = variant_pipeline.with_variant(select="add_div")
        >>> add_mul_pipeline(a=1, b=2, c=3)  # (1 + 2) * 3 = 9
        9
        >>> add_div_pipeline(a=1, b=2, c=3)  # (1 + 2) / 3 = 1.0
        1.0

        Notes
        -----
        - The `is_identical_pipefunc` function is used to determine if two `PipeFunc`
          instances are identical.
        - If multiple pipelines contain the same function but with different variant
          information, the function will be included multiple times in the
          resulting `VariantPipeline`, each with its respective variant assignment.

        """
        if len(variant_pipeline) < 2:  # noqa: PLR2004
            msg = "At least 2 pipelines must be provided."
            raise ValueError(msg)
        all_funcs: list[list[PipeFunc]] = []
        variant_info: list[tuple[str | None, str]] = []

        for item in variant_pipeline:
            if len(item) == 3:  # noqa: PLR2004
                variant_group, variant, pipeline = item
            else:
                variant, pipeline = item
                variant_group = None
            all_funcs.append(pipeline.functions)
            variant_info.append((variant_group, variant))

        # Find common functions using is_identical_pipefunc
        common_funcs: list[PipeFunc] = []
        for func in all_funcs[0]:
            is_common = True
            for other_funcs in all_funcs[1:]:
                if not _pipefunc_in_list(func, other_funcs):
                    is_common = False
                    break
            if is_common and not _pipefunc_in_list(func, common_funcs):
                common_funcs.append(func)

        functions: list[PipeFunc] = common_funcs[:]

        # Add unique functions with variant information
        for i, funcs in enumerate(all_funcs):
            variant_group, variant = variant_info[i]
            # Create the variants parameter based on variant_group
            variants_param = {variant_group: variant} if variant_group is not None else variant

            unique_funcs = [
                func.copy(variant=variants_param)
                for func in funcs
                if not _pipefunc_in_list(func, common_funcs)
            ]
            functions.extend(unique_funcs)

        return cls(functions)

    def visualize(self, **kwargs: Any) -> Any:
        """Visualize the VariantPipeline with interactive variant selection.

        Parameters
        ----------
        kwargs
            Additional keyword arguments passed to the `pipefunc.Pipeline.visualize` method.

        Returns
        -------
            The output of the widget.

        """
        requires("ipywidgets", reason="show_progress", extras="ipywidgets")

        return _create_variant_selection_widget(
            self,
            _update_visualization,  # type: ignore[arg-type]
            **kwargs,
        )

    def _repr_mimebundle_(
        self,
        include: set[str] | None = None,
        exclude: set[str] | None = None,
    ) -> dict[str, str]:  # pragma: no cover
        """Display the VariantPipeline widget or a text representation.

        Also displays a rich table of information if `rich` is installed.
        """
        if is_running_in_ipynb() and is_installed("rich") and is_installed("ipywidgets"):
            widget = _create_variant_selection_widget(
                self,
                _update_repr_mimebundle,  # type: ignore[arg-type]
            )
            return widget._repr_mimebundle_(include=include, exclude=exclude)
        # Return a plaintext representation of the object
        return {"text/plain": repr(self)}

    def __getattr__(self, name: str) -> None:
        if name in Pipeline.__dict__:
            msg = (
                "This is a `VariantPipeline`, not a `Pipeline`."
                " Use `pipeline.with_variant(...)` to select a variant first."
                f" Then call `variant_pipeline.{name}` again."
            )
            raise AttributeError(msg)
        default_msg = f"'VariantPipeline' object has no attribute '{name}'"
        raise AttributeError(default_msg)


def _validate_variants_exist(
    variants_mapping: dict[str | None, set[str]],
    selection: dict[str | None, str],
) -> None:
    """Validate that the specified variants exist."""
    for group, variant_name in selection.items():
        if group not in variants_mapping:
            msg = f"Unknown variant group: `{group}`."
            if variants_mapping:
                groups = (str(k) for k in variants_mapping)
                msg += f" Use one of: `{', '.join(groups)}`"
            raise ValueError(msg)
        if variant_name not in variants_mapping[group]:
            msg = (
                f"Unknown variant: `{variant_name}` in group `{group}`."
                f" Use one of: `{', '.join(variants_mapping[group])}`"
            )
            raise ValueError(msg)


def _pipefunc_in_list(func: PipeFunc, funcs: list[PipeFunc]) -> bool:
    """Check if a PipeFunc instance is in a list of PipeFunc instances."""
    return any(is_identical_pipefunc(func, f) for f in funcs)


def is_identical_pipefunc(first: PipeFunc, second: PipeFunc) -> bool:
    """Check if two PipeFunc instances are identical.

    Note: This is not implemented as PipeFunc.__eq__ to avoid
    hashing issues.
    """
    cls = type(first)
    for attr, value in first.__dict__.items():
        if isinstance(getattr(cls, attr, None), functools.cached_property):
            continue
        if attr == "_pipelines":
            continue
        if value != second.__dict__[attr]:
            return False
    return True


def _create_variant_selection_widget(
    vp: VariantPipeline,
    update_func: Callable[[Pipeline, ipywidgets.Output, Any], None],
    **kwargs: Any,
) -> ipywidgets.VBox:
    """Create a widget for interactive variant selection.

    Parameters
    ----------
    vp
        The VariantPipeline.
    update_func
        The function to call when the selected variant changes.
    kwargs
        Additional keyword arguments passed to the `Pipeline.visualize` method
        in `update_visualization`.

    Returns
    -------
        A widget containing dropdown menus for variant selection.

    """
    import ipywidgets

    dropdowns: dict[str | None, ipywidgets.Dropdown] = {}
    output = ipywidgets.Output()
    default = _ensure_dict(vp.default_variant)

    def wrapped_update_func(_change: dict | None = None) -> None:
        """Update the output with the selected variants."""
        selected_variants = {group: dropdowns[group].value for group in vp.variants_mapping()}
        pipeline = vp.with_variant(select=selected_variants)
        assert isinstance(pipeline, Pipeline)
        update_func(pipeline, output, **kwargs)  # type: ignore[call-arg]

    for group, variants in vp.variants_mapping().items():
        options = list(variants)
        dropdown = ipywidgets.Dropdown(
            options=options,
            value=default.get(group, options[0]),
            description=f"{group}:",
            disabled=False,
        )
        dropdown.observe(wrapped_update_func, names="value")
        dropdowns[group] = dropdown

    # Initial update
    wrapped_update_func()

    return ipywidgets.VBox([*dropdowns.values(), output])


def _ensure_dict(default_variant: str | dict[str | None, str] | None) -> dict[str | None, str]:
    """Ensure that the default_variant is a dictionary."""
    if default_variant is None:
        return {}
    if isinstance(default_variant, str):
        return {None: default_variant}
    return default_variant


def _update_visualization(
    pipeline: Pipeline,
    output: ipywidgets.Output,
    **kwargs: Any,
) -> None:
    """Update the visualization with the selected variants."""
    from IPython.display import display

    with output:
        output.clear_output()
        backend = kwargs.pop("backend", "graphviz")
        viz = pipeline.visualize(backend=backend, **kwargs)
        if viz is not None:
            display(viz)


def _update_repr_mimebundle(
    pipeline: Pipeline,
    output: ipywidgets.Output,
    **kwargs: Any,
) -> None:  # pragma: no cover
    """Update the displayed output with the selected variant's mimebundle."""
    from IPython.display import display

    with output:
        output.clear_output(wait=True)
        display(pipeline, **kwargs)
