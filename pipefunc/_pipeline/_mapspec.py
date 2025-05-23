from __future__ import annotations

from typing import TYPE_CHECKING

from pipefunc._utils import at_least_tuple
from pipefunc.map._mapspec import ArraySpec, MapSpec

if TYPE_CHECKING:
    from pipefunc._pipefunc import PipeFunc


def _axes_from_dims(p: str, dims: dict[str, int], axis: str) -> tuple[str | None, ...]:
    n = dims.get(p, 1) - 1
    return n * (None,) + (axis,)


def add_mapspec_axis(p: str, dims: dict[str, int], axis: str, functions: list[PipeFunc]) -> None:
    # Modify the MapSpec of functions that depend on `p` to include the new axis
    if "," in axis:
        # If the axis is a comma-separated list of axes, add each axis separately.
        for _axis in axis.split(","):
            _axis = _axis.strip()
            add_mapspec_axis(p, dims, _axis, functions)
        return

    for f in functions:
        if p not in f.parameters or p in f._bound:
            continue
        if f.mapspec is None:
            axes = _axes_from_dims(p, dims, axis)
            input_specs = [ArraySpec(p, axes)]
            output_specs = [ArraySpec(name, (axis,)) for name in at_least_tuple(f.output_name)]
        else:
            existing_inputs = set(f.mapspec.input_names)
            if p in existing_inputs:
                input_specs = [
                    s.add_axes(axis) if s.name == p and axis not in s.axes else s
                    for s in f.mapspec.inputs
                ]
            else:
                axes = _axes_from_dims(p, dims, axis)
                input_specs = [*f.mapspec.inputs, ArraySpec(p, axes)]
            output_specs = [
                s.add_axes(axis) if axis not in s.axes else s for s in f.mapspec.outputs
            ]
        f.mapspec = MapSpec(tuple(input_specs), tuple(output_specs), _is_generated=True)
        for o in output_specs:
            dims[o.name] = len(o.axes)
            add_mapspec_axis(o.name, dims, axis, functions)


def find_non_root_axes(
    mapspecs: list[MapSpec],
    root_args: list[str],
) -> dict[str, list[str | None]]:
    non_root_inputs: dict[str, list[str | None]] = {}
    for mapspec in mapspecs:
        for spec in mapspec.inputs:
            if spec.name not in root_args:
                if spec.name not in non_root_inputs:
                    non_root_inputs[spec.name] = spec.rank * [None]  # type: ignore[assignment]
                for i, axis in enumerate(spec.axes):
                    if axis is not None:
                        non_root_inputs[spec.name][i] = axis
    return non_root_inputs


def replace_none_in_axes(
    mapspecs: list[MapSpec],
    non_root_inputs: dict[str, list[str]],
    multi_output_mapping: dict[str, tuple[str, ...]],
) -> None:
    """Replaces `None` in the axes of non-root inputs with unique names.

    Mutates `non_root_inputs` in place!

    This sets the axes that are None to `unnamed_{i}` even though in
    a previous output the axis might have been named. This is not a problem
    because in that case this axis name won't be used. For example given
    `"x[i], y[j] -> a[i, j]"` and `"a[:, j] -> c[j]"` will still result in
    `{"a": ["unnamed_0", "j"]`.
    """
    all_axes_names = {
        axis.name for mapspec in mapspecs for axis in mapspec.inputs + mapspec.outputs
    }

    i = 0
    axis_template = "unnamed_{}"
    for name, axes in non_root_inputs.items():
        for j, axis in enumerate(axes):
            if axis is None:
                while (new_axis := axis_template.format(i)) in all_axes_names:
                    i += 1
                non_root_inputs[name][j] = new_axis
                all_axes_names.add(new_axis)
                if name in multi_output_mapping:
                    # If output is a tuple, update its axes with the new axis.
                    for output_name in multi_output_mapping[name]:
                        if output_name not in non_root_inputs:
                            continue
                        non_root_inputs[output_name][j] = new_axis
    assert not any(None in axes for axes in non_root_inputs.values())


def create_missing_mapspecs(
    functions: list[PipeFunc],
    non_root_inputs: dict[str, set[str]],
) -> set[PipeFunc]:
    # Mapping from output_name to PipeFunc for functions without a MapSpec
    outputs_without_mapspec: dict[str, PipeFunc] = {
        name: func
        for func in functions
        if func.mapspec is None
        for name in at_least_tuple(func.output_name)
    }

    missing: set[str] = non_root_inputs.keys() & outputs_without_mapspec.keys()
    func_with_new_mapspecs = set()
    for p in missing:
        func = outputs_without_mapspec[p]
        if func in func_with_new_mapspecs:
            continue  # already added a MapSpec because of multiple outputs
        axes = tuple(non_root_inputs[p])
        outputs = tuple(ArraySpec(x, axes) for x in at_least_tuple(func.output_name))
        func.mapspec = MapSpec(inputs=(), outputs=outputs, _is_generated=True)
        func_with_new_mapspecs.add(func)
        print(f"Autogenerated MapSpec for `{func}`: `{func.mapspec}`")
    return func_with_new_mapspecs
