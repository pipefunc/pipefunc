from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

from pipefunc import PipeFunc, Pipeline
from pipefunc._utils import assert_complete_kwargs


class VariantPipeline:
    """A pipeline container that supports multiple implementations (variants) of functions.

    `VariantPipeline` allows you to define multiple implementations of functions and
    select which variant to use at runtime. This is particularly useful for:

    - A/B testing different implementations
    - Experimenting with algorithm variations
    - Managing multiple processing options
    - Creating configurable pipelines

    The pipeline can have multiple variant groups, where each group contains alternative
    implementations of a function. Functions can be assigned to a variant group using
    the ``variant_group`` parameter and identified within that group using the ``variant``
    parameter.

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

        >>> @pipefunc(output_name="c", variant_group="method", variant="add")
        ... def f1(a, b):
        ...     return a + b
        ...
        >>> @pipefunc(output_name="c", variant_group="method", variant="sub")
        ... def f2(a, b):
        ...     return a - b
        ...
        >>> @pipefunc(output_name="d", variant_group="analysis", variant="mul")
        ... def g1(b, c):
        ...     return b * c
        ...
        >>> @pipefunc(output_name="d", variant_group="analysis", variant="div")
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
            if function.variant is None:
                assert function.variant_group is None
                continue
            variants = variant_groups.setdefault(function.variant_group, set())
            variants.add(function.variant)
        return variant_groups

    def _variants_mapping_inverse(self) -> dict[str, set[str | None]]:
        """Return a dictionary of variants and their variant groups."""
        variants: dict[str, set[str | None]] = {}
        for function in self.functions:
            if function.variant is None:
                assert function.variant_group is None
                continue
            groups = variants.setdefault(function.variant, set())
            groups.add(function.variant_group)
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
        """Resolve a single variant string to a dictionary.

        Raises:
            ValueError: If the variant is ambiguous or unknown.

        """
        inv = self._variants_mapping_inverse()
        group = inv.get(select, set())
        if len(group) > 1:
            msg = f"Ambiguous variant: `{select}`, could be in either `{group}`"
            raise ValueError(msg)
        if not group:
            msg = f"Unknown variant: `{select}`"
            raise ValueError(msg)
        return {group.pop(): select}

    def _select_functions(
        self,
        select: dict[str | None, str],
    ) -> list[PipeFunc]:
        """Select functions based on the given variant selection."""
        new_functions: list[PipeFunc] = []
        for function in self.functions:
            if function.variant is None:
                new_functions.append(function)
                continue
            if function.variant_group in select:
                if function.variant == select[function.variant_group]:
                    new_functions.append(function)
                else:
                    continue
            else:
                new_functions.append(function)
        return new_functions

    def _check_remaining_variants(self, functions: list[PipeFunc]) -> bool:
        """Check if any variants remain after selection."""
        left_over = defaultdict(set)
        for function in functions:
            if function.variant is not None:
                left_over[function.variant_group].add(function.variant)
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
