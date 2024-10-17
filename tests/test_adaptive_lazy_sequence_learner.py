from pipefunc.map._adaptive_lazy_sequence_learner import LazySequence, LazySequenceLearner

offset = 0.0123


def peak(x, offset=offset):
    a = 0.01
    return {"x": x + a**2 / (a**2 + (x - offset) ** 2)}


def test_lazy_sequence_learner():
    lazy_sequence = LazySequence(callable=lambda: [1, 2, 3, 4, 5])
    learner = LazySequenceLearner(peak, lazy_sequence)
    loss_improvement = 1 / 5
    assert learner.ask(1) == ([(0, 1)], [loss_improvement])
    assert learner.ask(1) == ([(1, 2)], [loss_improvement])
    assert learner.ask(1) == ([(2, 3)], [loss_improvement])
    assert learner.ask(2) == ([(3, 4), (4, 5)], [loss_improvement, loss_improvement])

    # Test __getstate__/__setstate__
    state = learner.__getstate__()
    learner2 = learner.new()
    learner2.__setstate__(state)
    assert learner2.ask(1) == ([(0, 1)], [loss_improvement])
    assert learner2.ask(1) == ([(1, 2)], [loss_improvement])
    assert learner2.ask(1) == ([(2, 3)], [loss_improvement])
