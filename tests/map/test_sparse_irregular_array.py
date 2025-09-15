"""Comprehensive tests for SparseIrregularArray."""

import numpy as np

from pipefunc.map._storage_array._sparse import SparseIrregularArray, SparseMask


class TestSparseIrregularArray:
    """Test suite for SparseIrregularArray."""

    def test_initialization_basic(self):
        """Test basic initialization."""
        data = {(0,): [1, 2, 3], (1,): [4], (2,): [5, 6]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(3,),
            shape_mask=(True, False),
        )

        assert sparse.shape == (3,)
        assert sparse.internal_shape == (3,)
        assert sparse.full_shape == (3, 3)
        assert sparse.ndim == 2
        assert sparse.dtype == np.dtype("O")

    def test_initialization_without_internal_shape(self):
        """Test initialization without internal shape (regular array)."""
        data = {(0,): 1, (1,): 2, (2,): 3}
        sparse = SparseIrregularArray(data_dict=data, shape=(3,), internal_shape=None)

        assert sparse.shape == (3,)
        assert sparse.internal_shape is None
        assert sparse.full_shape == (3,)
        assert sparse.ndim == 1

    def test_getitem_single_element(self):
        """Test accessing single elements."""
        data = {(0,): [1, 2, 3], (1,): [4], (2,): [5, 6]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(3,),
            shape_mask=(True, False),
        )

        # Valid accesses
        assert sparse[0, 0] == 1
        assert sparse[0, 1] == 2
        assert sparse[0, 2] == 3
        assert sparse[1, 0] == 4
        assert sparse[2, 0] == 5
        assert sparse[2, 1] == 6

        # Out of bounds - should return masked
        assert sparse[1, 1] is np.ma.masked
        assert sparse[1, 2] is np.ma.masked
        assert sparse[2, 2] is np.ma.masked

    def test_getitem_missing_data(self):
        """Test accessing missing data points."""
        data = {(0,): [1, 2], (2,): [3]}  # Missing (1,)
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )

        assert sparse[0, 0] == 1
        assert sparse[1, 0] is np.ma.masked  # Missing row
        assert sparse[1, 1] is np.ma.masked  # Missing row
        assert sparse[2, 0] == 3

    def test_iteration(self):
        """Test iteration over first dimension."""
        data = {(0,): [1, 2], (1,): [], (2,): [3, 4, 5]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(5,),
            shape_mask=(True, False),
        )

        rows = list(sparse)
        assert len(rows) == 3

        # First row: [1, 2, masked, masked, masked]
        assert rows[0][0] == 1
        assert rows[0][1] == 2
        assert rows[0].mask[2:].all()

        # Second row: all masked (empty)
        assert rows[1].mask.all()

        # Third row: [3, 4, 5, masked, masked]
        assert rows[2][0] == 3
        assert rows[2][1] == 4
        assert rows[2][2] == 5
        assert rows[2].mask[3:].all()

    def test_size_and_count(self):
        """Test size and count properties."""
        data = {(0,): [1, 2, 3], (1,): [4], (2,): []}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(5,),
            shape_mask=(True, False),
        )

        assert sparse.size == 15  # 3 * 5
        assert sparse.count() == 4  # 3 + 1 + 0 non-masked elements

    def test_compressed(self):
        """Test compressed method returns only non-masked values."""
        data = {(0,): [1, 2], (1,): [3], (2,): [4, 5, 6]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(4,),
            shape_mask=(True, False),
        )

        compressed = sparse.compressed()
        assert len(compressed) == 6
        assert list(compressed) == [1, 2, 3, 4, 5, 6]

    def test_to_dense_masked(self):
        """Test conversion to dense masked array."""
        data = {(0,): [1, 2], (2,): [3]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )

        dense = sparse.to_dense_masked()
        assert isinstance(dense, np.ma.MaskedArray)
        assert dense.shape == (3, 2)

        # Check values
        assert dense[0, 0] == 1
        assert dense[0, 1] == 2
        assert dense[2, 0] == 3

        # Check mask
        assert not dense.mask[0, 0]
        assert not dense.mask[0, 1]
        assert dense.mask[1, :].all()  # Row 1 all masked
        assert not dense.mask[2, 0]
        assert dense.mask[2, 1]

    def test_memory_efficiency(self):
        """Test memory efficiency for large sparse arrays."""
        # Create highly sparse data
        data = {(i,): [i] for i in range(0, 1000, 100)}  # Only 10 out of 1000 rows
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(1000,),
            internal_shape=(1000,),
            shape_mask=(True, False),
        )

        # Memory should be proportional to stored data, not full size
        assert sparse.nbytes < 1000  # Much less than 1000*1000*8 bytes
        assert sparse.count() == 10

        # Dense would be 1M elements
        assert sparse.size == 1_000_000

    def test_mask_property(self):
        """Test the mask property."""
        data = {(0,): [1, 2], (2,): [3]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )

        mask = sparse.mask
        assert isinstance(mask, SparseMask)
        assert mask.shape == (3, 2)

        # Test mask getitem
        assert not mask[0, 0]  # Has value
        assert not mask[0, 1]  # Has value
        assert mask[1, 0]  # Masked
        assert mask[1, 1]  # Masked
        assert not mask[2, 0]  # Has value
        assert mask[2, 1]  # Masked

    def test_array_conversion(self):
        """Test __array__ method for numpy compatibility."""
        data = {(0,): [1, 2], (1,): [3]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(2,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )

        # Should convert to numpy array
        arr = np.asarray(sparse)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (2, 2)

    def test_empty_arrays(self):
        """Test handling of empty arrays."""
        data = {(0,): [], (1,): [1], (2,): []}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )

        # Empty arrays should be all masked
        assert sparse[0, 0] is np.ma.masked
        assert sparse[0, 1] is np.ma.masked
        assert sparse[1, 0] == 1
        assert sparse[1, 1] is np.ma.masked
        assert sparse[2, 0] is np.ma.masked
        assert sparse[2, 1] is np.ma.masked

    def test_repr(self):
        """Test string representation."""
        data = {(0,): [1, 2], (1,): [3]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(2,),
            internal_shape=(3,),
            shape_mask=(True, False),
        )

        repr_str = repr(sparse)
        assert "SparseIrregularArray" in repr_str
        assert "shape=(2, 3)" in repr_str
        assert "stored=2/6" in repr_str  # 2 stored out of 6 total
        assert "33.3%" in repr_str  # Density

    def test_edge_cases(self):
        """Test edge cases."""
        # Single element
        data = {(0,): [42]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(1,),
            internal_shape=(1,),
            shape_mask=(True, False),
        )
        assert sparse[0, 0] == 42
        assert sparse.count() == 1

        # Very large internal shape with single element
        data = {(0,): [1]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(1,),
            internal_shape=(1_000_000,),
            shape_mask=(True, False),
        )
        assert sparse[0, 0] == 1
        assert sparse[0, 999999] is np.ma.masked
        assert sparse.count() == 1
        assert sparse.size == 1_000_000

    def test_comparison_with_dense(self):
        """Test that sparse and dense representations are equivalent."""
        data = {(0,): [1, 2, 3], (1,): [4], (2,): [5, 6]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(3,),
            shape_mask=(True, False),
        )

        # Convert to dense
        dense = sparse.to_dense_masked()

        # Compare all elements
        for i in range(3):
            for j in range(3):
                sparse_val = sparse[i, j]
                dense_val = dense[i, j]
                if sparse_val is np.ma.masked:
                    assert dense.mask[i, j]
                else:
                    assert sparse_val == dense_val

    def test_slicing_fallback(self):
        """Test that slicing falls back to dense conversion."""
        data = {(0,): [1, 2], (1,): [3, 4, 5]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(2,),
            internal_shape=(5,),
            shape_mask=(True, False),
        )

        # Slicing should work (via dense conversion)
        sliced = sparse[:, :2]
        assert sliced.shape == (2, 2)
        assert sliced[0, 0] == 1
        assert sliced[0, 1] == 2
        assert sliced[1, 0] == 3
        assert sliced[1, 1] == 4

    def test_3d_array(self):
        """Test 3D sparse array."""
        # This tests the generalization beyond 2D
        data = {
            (0, 0): [1, 2],
            (0, 1): [3],
            (1, 0): [4, 5, 6],
        }
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(2, 2),
            internal_shape=(3,),
            shape_mask=(True, True, False),
        )

        assert sparse.full_shape == (2, 2, 3)
        assert sparse[0, 0, 0] == 1
        assert sparse[0, 0, 1] == 2
        assert sparse[0, 0, 2] is np.ma.masked
        assert sparse[0, 1, 0] == 3
        assert sparse[1, 0, 2] == 6
        assert sparse[1, 1, 0] is np.ma.masked  # Missing (1,1)


class TestSparseMask:
    """Test suite for SparseMask."""

    def test_mask_single_element(self):
        """Test mask for single elements."""
        data = {(0,): [1, 2], (2,): [3]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(3,),
            internal_shape=(2,),
            shape_mask=(True, False),
        )
        mask = sparse.mask

        assert not mask[0, 0]  # Has value
        assert not mask[0, 1]  # Has value
        assert mask[1, 0]  # No data for row 1
        assert mask[1, 1]  # No data for row 1
        assert not mask[2, 0]  # Has value
        assert mask[2, 1]  # Out of bounds for row 2

    def test_mask_shape(self):
        """Test mask shape property."""
        data = {(0,): [1]}
        sparse = SparseIrregularArray(
            data_dict=data,
            shape=(2,),
            internal_shape=(3,),
            shape_mask=(True, False),
        )

        assert sparse.mask.shape == (2, 3)
