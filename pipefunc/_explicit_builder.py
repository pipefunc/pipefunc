from __future__ import annotations
from typing import Callable
from pipefunc._pipefunc import PipeFunc

class PipelineBuilder:
    def add(self, node: Callable, **node_kwargs):
        if isinstance(node, Node):
        if node_kwargs and not isinstance(node, Node):
            node = PipeFunc(node)
        if isinstance(node, Node):
            ...
        else:
            ...
@dataclass
class Node:
    name:
    kwargs: dict[str, Node]