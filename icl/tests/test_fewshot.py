import numpy as np

from icl import fewshot


def clusters(n_classes=6, per_class=40, dim=24, spread=0.05, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(n_classes, dim))
    labels = np.repeat(np.arange(n_classes), per_class)
    return (means[labels] + spread * rng.normal(size=(len(labels), dim)) + 5.0).astype(np.float32), labels


def test_separable_features_are_classified_and_random_ones_are_not():
    feats, labels = clusters()
    f = fewshot.cl2n(feats, feats.mean(0))
    np.testing.assert_allclose(np.linalg.norm(f, axis=1), 1.0, atol=1e-5)
    good = fewshot.evaluate_pool(f, labels, "pool", n_way=6, shots=[1, 5, 10], n_query=15, n_episodes=200, seed=0)
    assert good["n_way"] == 6 and good["images"] == 240 and good["classes"] == 6
    for k in ("1", "5", "10"):
        assert good[k]["acc"] > 0.99 and good[k]["episodes"] == 200 and good[k]["n_query"] == 15
    noise = np.random.default_rng(1).normal(size=feats.shape).astype(np.float32)
    bad = fewshot.evaluate_pool(fewshot.cl2n(noise, noise.mean(0)), labels, "pool", n_way=6, shots=[5],
                                n_query=15, n_episodes=400, seed=0)["5"]
    assert abs(bad["acc"] - 1 / 6) < 3 * bad["ci95"] / 1.96 + 0.01 and 0 < bad["ci95"] < 0.02
    # without the offset removed, the common +5 dominates every direction and one shot gets much harder
    raw = fewshot.cl2n(feats, np.zeros(feats.shape[1], np.float32))
    assert np.abs(raw @ raw.T).min() > 0.9


def test_harder_features_are_in_between_and_more_shots_help():
    feats, labels = clusters(spread=1.5, seed=3)
    r = fewshot.evaluate_pool(fewshot.cl2n(feats, feats.mean(0)), labels, "pool", n_way=6, shots=[1, 10],
                              n_query=15, n_episodes=300, seed=0)
    assert 1 / 6 + 0.05 < r["1"]["acc"] < r["10"]["acc"] < 0.999


def test_episodes_are_fixed_by_seed_pool_name_and_shots():
    _, labels = clusters()
    a = fewshot.make_episodes(labels, 4, 5, 15, 50, seed=0, name="cats")
    b = fewshot.make_episodes(labels, 4, 5, 15, 50, seed=0, name="cats")
    assert a[2] == 15 and a[0].shape == (50, 4, 5) and a[1].shape == (50, 4, 15)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    for other in (fewshot.make_episodes(labels, 4, 5, 15, 50, seed=1, name="cats"),
                  fewshot.make_episodes(labels, 4, 5, 15, 50, seed=0, name="dogs")):
        assert not np.array_equal(a[0], other[0])
    sup, qry, _ = a
    for e in range(50):
        classes = labels[sup[e, :, 0]]
        assert len(set(classes.tolist())) == 4                                    # four different classes
        assert np.all(labels[sup[e]] == classes[:, None]) and np.all(labels[qry[e]] == classes[:, None])
        assert not set(sup[e].ravel().tolist()) & set(qry[e].ravel().tolist())    # no query is a support image
        assert len(set(qry[e].ravel().tolist())) == qry[e].size
    assert len({tuple(sorted(labels[sup[e, :, 0]].tolist())) for e in range(50)}) > 1   # classes are redrawn
    whole = fewshot.make_episodes(labels, 6, 1, 15, 5, seed=0, name="cats")
    assert all(labels[whole[0][e, :, 0]].tolist() == list(range(6)) for e in range(5))  # N = all: fixed order


def test_short_classes_shorten_the_queries_or_rule_a_setting_out():
    labels = np.array([0] * 12 + [1] * 30 + [2] * 30)
    sup, qry, q = fewshot.make_episodes(labels, 3, 5, 15, 10, seed=0, name="p")
    assert q == 7 and qry.shape == (10, 3, 7)
    assert fewshot.make_episodes(labels, 3, 12, 15, 10, seed=0, name="p") is None      # class 0 has no 13th image
    two = fewshot.make_episodes(labels, 2, 12, 15, 10, seed=0, name="p")               # two classes still do
    assert two[2] == 15 and set(labels[two[0]].ravel().tolist()) == {1, 2}
    f = np.eye(3, dtype=np.float32)[labels] + 0.01
    out = fewshot.evaluate_pool(fewshot.cl2n(f, f.mean(0)), labels, "p", n_way=3, shots=[5, 12], n_query=15,
                                n_episodes=10, seed=0)
    assert out["12"] is None and out["5"]["n_query"] == 7 and out["5"]["acc"] == 1.0


def test_episode_accuracy_matches_a_plain_loop():
    feats, labels = clusters(spread=1.2, seed=5)
    f = fewshot.cl2n(feats, feats.mean(0))
    sup, qry, q = fewshot.make_episodes(labels, 5, 3, 6, 7, seed=0, name="x")
    got = fewshot.episode_accuracy(f, sup, qry, chunk=3)
    for e in range(7):
        cent = np.stack([f[sup[e, j]].mean(0) for j in range(5)])
        hits = [int(np.argmin(((f[i] - cent) ** 2).sum(1)) == j) for j in range(5) for i in qry[e, j]]
        assert abs(got[e] - np.mean(hits)) < 1e-12
