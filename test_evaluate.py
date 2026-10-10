def is_duplicate(similarity, threshold=0.65):
    return similarity >= threshold


def test_high_similarity_is_duplicate():
    assert is_duplicate(0.80) is True


def test_low_similarity_is_not_duplicate():
    assert is_duplicate(0.40) is False


def test_similarity_at_threshold_is_duplicate():
    assert is_duplicate(0.65) is True