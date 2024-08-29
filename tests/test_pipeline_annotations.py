from pipefunc import Pipeline, pipefunc


def test_axis_is_reduced():
    @pipefunc("y", mapspec="x[i] -> y[i]")
    def f(x):
        return x

    @pipefunc("z")
    def g(y: list[int]) -> list[int]:
        return y

    Pipeline([f, g])
