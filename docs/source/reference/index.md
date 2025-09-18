# 📜 API Documentation

!!! tip "tl;dr"
    The API documentation is comprehensive and can be overwhelming.
    The most important parts are:

    - `pipefunc.pipefunc`: the `@pipefunc` decorator
    - `pipefunc.PipeFunc`: the class that is returned by the `@pipefunc` decorator
    - `pipefunc.Pipeline`: the class containing the `PipeFunc` instances
    - `pipefunc.Pipeline.run`: run functions inline sequentially
    - `pipefunc.Pipeline.map`: run functions that *may* contain map-reduce operations in parallel

## Modules

- [pipefunc](pipefunc.md)
- [pipefunc.map](pipefunc.map.md)
- [pipefunc.map.adaptive](pipefunc.map.adaptive.md)
- [pipefunc.map.xarray](pipefunc.map.xarray.md)
- [pipefunc.map.adaptive_scheduler](pipefunc.map.adaptive_scheduler.md)
- [pipefunc.cache](pipefunc.cache.md)
- [pipefunc.helpers](pipefunc.helpers.md)
- [pipefunc.resources](pipefunc.resources.md)
- [pipefunc.lazy](pipefunc.lazy.md)
- [pipefunc.mcp](pipefunc.mcp.md)
- [pipefunc.sweep](pipefunc.sweep.md)
- [pipefunc.testing](pipefunc.testing.md)
- [pipefunc.typing](pipefunc.typing.md)
