# 📜 API Documentation

```{admonition} tl;dr
The API documentation is comprehensive and can be overwhelming.
The most important parts are:

- {class}`pipefunc.pipefunc`: the ``@pipefunc`` decorator
- {class}`pipefunc.PipeFunc`: the class that is returned by the ``@pipefunc`` decorator
- {class}`pipefunc.Pipeline`: the class containing the ``PipeFunc`` instances
- {class}`pipefunc.Pipeline.run`: run functions inline sequentially
- {class}`pipefunc.Pipeline.map`: run functions that *may* contain map-reduce operations in parallel

```

```{toctree}
pipefunc
pipefunc.map
pipefunc.map.adaptive
pipefunc.map.xarray
pipefunc.map.adaptive_scheduler
pipefunc.cache
pipefunc.helpers
pipefunc.resources
pipefunc.lazy
pipefunc.sweep
pipefunc.testing
pipefunc.typing
```
