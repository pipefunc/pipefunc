"""Tests for DictArray with sparse irregular array support."""

import numpy as np

from pipefunc.map._storage_array._dict import DictArray
from pipefunc.map._storage_array._sparse import SparseIrregularArray


class TestDictArraySparse:
    """Test DictArray with sparse array support."""

    def test_to_array_sparse_auto_detect_large(self):
        """Test auto-detection of sparse for large arrays."""
        # Create a DictArray with irregular data
        dict_array = DictArray(
            folder=None,
            shape=(3,),
            internal_shape=(10000,),  # Large internal shape
            shape_mask=(True, False),
            irregular=True,
        )

        # Add sparse data
        dict_array.dump((0,), [1, 2, 3])
        dict_array.dump((1,), [4])
        dict_array.dump((2,), list(range(100)))  # Still small relative to 10000

        # Should auto-detect sparse is beneficial
        result = dict_array.to_array(sparse=None)
        assert isinstance(result, SparseIrregularArray)
        assert result.full_shape == (3, 10000)
        assert result.count() == 104  # 3 + 1 + 100

    def test_to_array_sparse_auto_detect_low_density(self):
        """Test auto-detection of sparse for low density arrays."""
        dict_array = DictArray(
            folder=None,
            shape=(100,),
            internal_shape=(100,),
            shape_mask=(True, False),
            irregular=True,
        )

        # Add very sparse data (only 5 out of 100 rows)
        for i in range(0, 100, 20):
            dict_array.dump((i,), [i])

        # Should auto-detect sparse is beneficial (5% density)
        result = dict_array.to_array(sparse=None)
        assert isinstance(result, SparseIrregularArray)
        assert result.count() == 5

    def test_to_array_sparse_explicit_true(self):
        """Test explicit sparse=True."""
        dict_array = DictArray(
            folder=None,
            shape=(2,),
            internal_shape=(3,),
            shape_mask=(True, False),
            irregular=True,
        )

        dict_array.dump((0,), [1, 2])
        dict_array.dump((1,), [3])

        result = dict_array.to_array(sparse=True)
        assert isinstance(result, SparseIrregularArray)
        assert result[0, 0] == 1
        assert result[0, 1] == 2
        assert result[0, 2] is np.ma.masked
        assert result[1, 0] == 3

    def test_to_array_sparse_explicit_false(self):
        """Test explicit sparse=False returns dense."""
        dict_array = DictArray(
            folder=None,
            shape=(2,),
            internal_shape=(2,),
            shape_mask=(True, False),
            irregular=True,
        )

        dict_array.dump((0,), [1, 2])
        dict_array.dump((1,), [3])

        result = dict_array.to_array(sparse=False)
        assert isinstance(result, np.ma.MaskedArray)
        assert not isinstance(result, SparseIrregularArray)

    def test_to_array_non_irregular_ignores_sparse(self):
        """Test that non-irregular arrays ignore sparse parameter."""
        dict_array = DictArray(
            folder=None,
            shape=(2,),
            internal_shape=(2,),
            shape_mask=(True, False),
            irregular=False,  # Not irregular
        )

        dict_array.dump((0,), [1, 2])
        dict_array.dump((1,), [3, 4])

        result = dict_array.to_array(sparse=True)
        # Should return regular MaskedArray, not sparse
        assert isinstance(result, np.ma.MaskedArray)
        assert not isinstance(result, SparseIrregularArray)

    def test_memory_comparison(self):
        """Test memory usage comparison between sparse and dense."""
        # Create dict array with one huge array among many small ones
        dict_array = DictArray(
            folder=None,
            shape=(1000,),
            internal_shape=(1000,),
            shape_mask=(True, False),
            irregular=True,
        )

        # 999 arrays with 1 element each
        for i in range(999):
            dict_array.dump((i,), [i])

        # One array with 1000 elements
        dict_array.dump((999,), list(range(1000)))

        # Get sparse representation
        sparse = dict_array.to_array(sparse=True)
        assert isinstance(sparse, SparseIrregularArray)

        # Memory should be much less than dense
        assert sparse.nbytes < 100_000  # Much less than 1M * 8 bytes

        # Dense representation would be huge
        assert sparse.size == 1_000_000

    def test_sparse_array_operations(self):
        """Test that sparse arrays support expected operations."""
        dict_array = DictArray(
            folder=None,
            shape=(3,),
            internal_shape=(5,),
            shape_mask=(True, False),
            irregular=True,
        )

        dict_array.dump((0,), [1, 2, 3])
        dict_array.dump((1,), [])
        dict_array.dump((2,), [4, 5])

        sparse = dict_array.to_array(sparse=True)

        # Test iteration
        rows = list(sparse)
        assert len(rows) == 3
        assert rows[0][0] == 1
        assert rows[1].mask.all()  # Empty row

        # Test compressed
        compressed = sparse.compressed()
        assert list(compressed) == [1, 2, 3, 4, 5]

        # Test count
        assert sparse.count() == 5

    def test_sparse_with_missing_keys(self):
        """Test sparse arrays with missing dictionary keys."""
        dict_array = DictArray(
            folder=None,
            shape=(5,),
            internal_shape=(2,),
            shape_mask=(True, False),
            irregular=True,
        )

        # Only add data for some indices
        dict_array.dump((0,), [1, 2])
        dict_array.dump((3,), [3])
        # Indices 1, 2, 4 are missing

        sparse = dict_array.to_array(sparse=True)
        assert sparse[0, 0] == 1
        assert sparse[0, 1] == 2
        assert sparse[1, 0] is np.ma.masked  # Missing key
        assert sparse[2, 0] is np.ma.masked  # Missing key
        assert sparse[3, 0] == 3
        assert sparse[4, 0] is np.ma.masked  # Missing key

    def test_sparse_3d(self):
        """Test 3D sparse arrays from DictArray."""
        dict_array = DictArray(
            folder=None,
            shape=(2, 2),
            internal_shape=(3,),
            shape_mask=(True, True, False),
            irregular=True,
        )

        dict_array.dump((0, 0), [1, 2])
        dict_array.dump((0, 1), [3])
        dict_array.dump((1, 0), [4, 5, 6])

        sparse = dict_array.to_array(sparse=True)
        assert sparse.full_shape == (2, 2, 3)
        assert sparse[0, 0, 0] == 1
        assert sparse[0, 0, 2] is np.ma.masked
        assert sparse[1, 0, 2] == 6

    def test_equivalence_sparse_dense(self):
        """Test that sparse and dense representations are equivalent."""
        dict_array = DictArray(
            folder=None,
            shape=(3,),
            internal_shape=(4,),
            shape_mask=(True, False),
            irregular=True,
        )

        dict_array.dump((0,), [1, 2, 3])
        dict_array.dump((1,), [4])
        dict_array.dump((2,), [5, 6])

        # Get both representations
        sparse = dict_array.to_array(sparse=True)
        dense = dict_array.to_array(sparse=False)

        # Compare all elements
        for i in range(3):
            for j in range(4):
                sparse_val = sparse[i, j]
                dense_val = dense[i, j]
                if sparse_val is np.ma.masked:
                    assert dense.mask[i, j]
                else:
                    assert sparse_val == dense_val

    def test_auto_detect_small_dense(self):
        """Test that small dense arrays don't use sparse."""
        dict_array = DictArray(
            folder=None,
            shape=(5,),
            internal_shape=(5,),
            shape_mask=(True, False),
            irregular=True,
        )

        # Fill most of the array
        for i in range(5):
            dict_array.dump((i,), list(range(4)))  # 80% filled

        # Should not auto-detect sparse (high density, small size)
        result = dict_array.to_array(sparse=None)
        assert isinstance(result, np.ma.MaskedArray)
        assert not isinstance(result, SparseIrregularArray)
