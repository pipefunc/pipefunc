"""End-to-end tests for irregular arrays with sparse support in pipefunc pipelines."""

import numpy as np

from pipefunc import Pipeline, pipefunc
from pipefunc.map._storage_array._sparse import SparseIrregularArray


class TestIrregularSparsePipeline:
    """Test irregular arrays with sparse support in full pipelines."""

    def test_highly_variable_sizes(self):
        """Test pipeline with highly variable array sizes."""

        @pipefunc(output_name="data", mapspec="size[i] -> data[i, j*]")
        def generate_data(size: int) -> list[int]:
            """Generate data with wildly different sizes."""
            return list(range(size))

        @pipefunc(output_name="squared", mapspec="data[i, j*] -> squared[i, j*]")
        def square_data(data: int) -> int:
            """Square each element."""
            if data is np.ma.masked:
                return np.ma.masked  # type: ignore[return-value]
            return data**2

        @pipefunc(output_name="sum", mapspec="squared[i, :] -> sum[i]")
        def sum_row(squared: np.ndarray) -> int:
            """Sum the row."""
            if hasattr(squared, "compressed"):
                return sum(squared.compressed())
            total = 0
            for val in squared:
                if val is not np.ma.masked:
                    total += val
            return total

        pipeline = Pipeline([generate_data, square_data, sum_row])

        # Create inputs with extreme size variation
        # Most are tiny, one is huge
        sizes = [1] * 99 + [1000]  # 99 arrays of size 1, 1 array of size 1000

        results = pipeline.map(
            inputs={"size": sizes},
            internal_shapes={"data": (1000,), "squared": (1000,)},
            parallel=False,
            storage="dict",
        )

        # Check that data is stored as sparse
        data_store = results["data"].store
        data_array = data_store.to_array(sparse=True)  # Request sparse explicitly

        # Should be sparse array for efficiency
        assert isinstance(data_array, SparseIrregularArray)
        assert data_array.full_shape == (100, 1000)

        # Check memory efficiency
        # Should use much less memory than 100*1000*8 bytes
        assert data_array.nbytes < 100_000  # Much less than 800KB

        # Check results are correct
        sum_array = results["sum"].output
        assert sum_array[0] == 0  # sum([0^2]) = 0
        for i in range(1, 99):
            assert sum_array[i] == 0  # All small arrays sum to 0

        # Large array: sum of squares from 0 to 999
        expected_sum = sum(i**2 for i in range(1000))
        assert sum_array[99] == expected_sum

    def test_sparse_with_empty_arrays(self):
        """Test sparse arrays with some empty entries."""

        @pipefunc(output_name="chars", mapspec="text[i] -> chars[i, j*]")
        def text_to_chars(text: str) -> list[str]:
            """Convert text to list of characters."""
            return list(text)

        @pipefunc(output_name="is_vowel", mapspec="chars[i, j*] -> is_vowel[i, j*]")
        def check_vowel(chars: str) -> bool:
            """Check if character is a vowel."""
            if chars is np.ma.masked:
                return np.ma.masked  # type: ignore[return-value]
            return chars.lower() in "aeiou"

        @pipefunc(output_name="vowel_count", mapspec="is_vowel[i, :] -> vowel_count[i]")
        def count_vowels(is_vowel: np.ndarray) -> int:
            """Count vowels."""
            if hasattr(is_vowel, "compressed"):
                return sum(is_vowel.compressed())
            count = 0
            for val in is_vowel:
                if val is not np.ma.masked and val:
                    count += 1
            return count

        pipeline = Pipeline([text_to_chars, check_vowel, count_vowels])

        # Mix of empty strings, short strings, and one long string
        texts = [""] * 10 + ["Hi"] * 10 + ["A" * 1000]  # 10 empty, 10 short, 1 long

        results = pipeline.map(
            inputs={"text": texts},
            internal_shapes={"chars": (1000,), "is_vowel": (1000,)},
            parallel=False,
            storage="dict",
        )

        # Check sparse storage
        chars_store = results["chars"].store
        chars_array = chars_store.to_array(sparse=True)

        assert isinstance(chars_array, SparseIrregularArray)
        assert chars_array.count() == 10 * 2 + 1000  # 10 * "Hi" + 1000 * "A"

        # Check results
        vowel_counts = results["vowel_count"].output
        assert all(vowel_counts[:10] == 0)  # Empty strings have 0 vowels
        assert all(vowel_counts[10:20] == 1)  # "Hi" has 1 vowel
        assert vowel_counts[20] == 1000  # All A's are vowels

    def test_auto_sparse_detection(self):
        """Test that sparse is automatically used when beneficial."""

        @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
        def generate(n: int) -> list[float]:
            """Generate n floats."""
            return [float(i) for i in range(n)]

        pipeline = Pipeline([generate])

        # Large internal shape should trigger auto-sparse
        results = pipeline.map(
            inputs={"n": [1, 2, 3]},
            internal_shapes={"data": (100000,)},  # Large internal shape
            parallel=False,
            storage="dict",
        )

        # Check that sparse was auto-selected
        data_store = results["data"].store
        # Don't explicitly request sparse, let it auto-detect
        data_array = data_store.to_array()

        # Should auto-detect sparse is beneficial
        assert isinstance(data_array, SparseIrregularArray)

    def test_sparse_with_parallel_execution(self):
        """Test sparse arrays work with parallel execution."""

        @pipefunc(output_name="data", mapspec="size[i] -> data[i, j*]")
        def generate_data(size: int) -> list[int]:
            """Generate data."""
            return list(range(size))

        pipeline = Pipeline([generate_data])

        # Various sizes
        sizes = [1, 10, 100, 1000]

        results = pipeline.map(
            inputs={"size": sizes},
            internal_shapes={"data": (1000,)},
            parallel=True,  # Use parallel execution
            storage="shared_memory_dict",
        )

        # Even with parallel execution, sparse should work
        data_store = results["data"].store
        data_array = data_store.to_array(sparse=True)

        assert isinstance(data_array, SparseIrregularArray)
        assert data_array.count() == 1 + 10 + 100 + 1000

    def test_sparse_dense_equivalence(self):
        """Test that sparse and dense give same results."""

        @pipefunc(output_name="lists", mapspec="n[i] -> lists[i, j*]")
        def make_lists(n: int) -> list[int]:
            """Create lists."""
            return [x * n for x in range(n)]

        @pipefunc(output_name="doubled", mapspec="lists[i, j*] -> doubled[i, j*]")
        def double(lists: int) -> int:
            """Double values."""
            if lists is np.ma.masked:
                return np.ma.masked  # type: ignore[return-value]
            return lists * 2

        pipeline = Pipeline([make_lists, double])

        inputs = {"n": [1, 2, 3, 4, 5]}

        results = pipeline.map(
            inputs=inputs,
            internal_shapes={"lists": (5,), "doubled": (5,)},
            parallel=False,
            storage="dict",
        )

        # Get both sparse and dense representations
        doubled_store = results["doubled"].store
        sparse = doubled_store.to_array(sparse=True)
        dense = doubled_store.to_array(sparse=False)

        # Should give same values (where both have data)
        # Note: sparse and dense may handle the "doubled" values differently
        # The important thing is they have the same actual values
        assert sparse.count() == np.ma.count(dense)

        # Compare the compressed (non-masked) values
        sparse_compressed = sorted(sparse.compressed().tolist())
        dense_compressed = sorted(dense.compressed().tolist())
        assert sparse_compressed == dense_compressed

    def test_memory_scaling(self):
        """Test that memory usage scales with actual data, not maximum size."""

        @pipefunc(output_name="data", mapspec="size[i] -> data[i, j*]")
        def generate(size: int) -> list[int]:
            """Generate data."""
            return list(range(size))

        pipeline = Pipeline([generate])

        # Test with different maximum sizes but same actual data
        for max_size in [1000, 10000, 100000]:
            results = pipeline.map(
                inputs={"size": [10, 20, 30]},  # Same actual sizes
                internal_shapes={"data": (max_size,)},  # Different max sizes
                parallel=False,
                storage="dict",
            )

            data_store = results["data"].store
            sparse = data_store.to_array(sparse=True)

            # Memory should be proportional to actual data (60 elements)
            # not to max_size
            assert sparse.nbytes < 1000  # Should be ~480 bytes
            assert sparse.count() == 60  # Same actual data

    def test_sparse_with_reductions(self):
        """Test that reductions work correctly with sparse arrays."""

        @pipefunc(output_name="data", mapspec="n[i] -> data[i, j*]")
        def generate(n: int) -> list[int]:
            """Generate data."""
            return list(range(1, n + 1))

        @pipefunc(output_name="product", mapspec="data[i, :] -> product[i]")
        def compute_product(data: np.ndarray) -> int:
            """Compute product of all values."""
            if hasattr(data, "compressed"):
                result = 1
                for val in data.compressed():
                    result *= val
                return result
            result = 1
            for val in data:
                if val is not np.ma.masked:
                    result *= val
            return result

        pipeline = Pipeline([generate, compute_product])

        results = pipeline.map(
            inputs={"n": [0, 1, 2, 3, 4]},
            internal_shapes={"data": (10,)},
            parallel=False,
            storage="dict",
        )

        products = results["product"].output
        assert products[0] == 1  # Empty product is 1
        assert products[1] == 1  # 1
        assert products[2] == 2  # 1 * 2
        assert products[3] == 6  # 1 * 2 * 3
        assert products[4] == 24  # 1 * 2 * 3 * 4
