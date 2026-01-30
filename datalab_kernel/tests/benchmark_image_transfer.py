# Copyright (c) DataLab Platform Developers, BSD 3-Clause License
# See LICENSE file for details

"""
Benchmark for Image Transfer Performance
=========================================

This script benchmarks the performance of image object serialization
and deserialization, as well as simulating the byte conversion overhead
that occurs in Pyodide/JupyterLite environments.

The main bottleneck in Pyodide is:
1. Response byte conversion: `bytes([ord(c) & 0xFF for c in response_text])`
2. Request byte upload: byte-by-byte Uint8Array population

These are O(n) Python loops that become very slow for large images.

Run with::

    python scripts/run_with_env.py python -m pytest \
        datalab_kernel/tests/benchmark_image_transfer.py -v -s
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from sigima import ImageObj

from datalab_kernel.serialization_npz import (
    deserialize_object_from_npz,
    serialize_object_to_npz,
)


def create_test_image(size: int, dtype=np.float32) -> ImageObj:
    """Create a test image of given size."""
    data = np.random.rand(size, size).astype(dtype)
    obj = ImageObj()
    obj.data = data
    obj.title = f"Test Image {size}x{size}"
    return obj


def simulate_pyodide_response_conversion(data: bytes) -> bytes:
    """Simulate the current Pyodide response byte conversion.

    This is the bottleneck: `bytes([ord(c) & 0xFF for c in response_text])`
    In reality, response_text is a string, but we simulate with bytes.
    """
    # Simulate the string conversion that happens in browser
    response_text = data.decode("latin-1")  # This preserves all byte values
    # This is the slow part - O(n) Python loop
    return bytes([ord(c) & 0xFF for c in response_text])


def simulate_pyodide_request_preparation(data: bytes) -> list[int]:
    """Simulate the current Pyodide request byte preparation.

    This simulates populating a Uint8Array byte-by-byte.
    """
    # This is the slow part - O(n) Python loop
    return list(data)


def optimized_response_conversion(data: bytes) -> bytes:
    """Optimized response byte conversion using NumPy.

    Instead of Python loop, use NumPy for vectorized conversion.
    """
    response_text = data.decode("latin-1")
    # Use NumPy's frombuffer for efficient conversion
    return np.frombuffer(response_text.encode("latin-1"), dtype=np.uint8).tobytes()


def optimized_request_preparation_memoryview(data: bytes) -> memoryview:
    """Optimized request preparation using memoryview.

    Returns a memoryview that can be efficiently passed to JavaScript.
    """
    return memoryview(data)


class TestImageSerializationPerformance:
    """Benchmark tests for image serialization performance."""

    IMAGE_SIZES = [256, 512, 1024, 2048]

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    def test_serialization_speed(self, size: int):
        """Benchmark NPZ serialization speed for different image sizes."""
        obj = create_test_image(size)
        data_size_mb = (size * size * 4) / (1024 * 1024)  # float32 = 4 bytes

        # Warm up
        serialize_object_to_npz(obj)

        # Benchmark serialization
        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            npz_data = serialize_object_to_npz(obj)
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        compressed_size_mb = len(npz_data) / (1024 * 1024)
        compression_ratio = data_size_mb / compressed_size_mb

        print(
            f"\n[Serialize {size}x{size}] Raw: {data_size_mb:.2f} MB, "
            f"Compressed: {compressed_size_mb:.2f} MB "
            f"(ratio: {compression_ratio:.1f}x), "
            f"Time: {avg_time * 1000:.1f} ms"
        )

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    def test_deserialization_speed(self, size: int):
        """Benchmark NPZ deserialization speed."""
        obj = create_test_image(size)
        npz_data = serialize_object_to_npz(obj)

        # Warm up
        deserialize_object_from_npz(npz_data)

        # Benchmark deserialization
        start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            deserialize_object_from_npz(npz_data)
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations

        print(f"\n[Deserialize {size}x{size}] Time: {avg_time * 1000:.1f} ms")

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    def test_pyodide_response_conversion_bottleneck(self, size: int):
        """Benchmark the Pyodide response byte conversion bottleneck."""
        obj = create_test_image(size)
        npz_data = serialize_object_to_npz(obj)
        data_size_mb = len(npz_data) / (1024 * 1024)

        # Current slow method
        start = time.perf_counter()
        _ = simulate_pyodide_response_conversion(npz_data)
        slow_time = time.perf_counter() - start

        # Optimized method
        start = time.perf_counter()
        _ = optimized_response_conversion(npz_data)
        fast_time = time.perf_counter() - start

        speedup = slow_time / fast_time if fast_time > 0 else float("inf")

        print(
            f"\n[Response Conversion {size}x{size}] "
            f"Data: {data_size_mb:.2f} MB, "
            f"Slow: {slow_time * 1000:.1f} ms, "
            f"Fast: {fast_time * 1000:.1f} ms, "
            f"Speedup: {speedup:.1f}x"
        )

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    def test_pyodide_request_preparation_bottleneck(self, size: int):
        """Benchmark the Pyodide request byte preparation bottleneck."""
        obj = create_test_image(size)
        npz_data = serialize_object_to_npz(obj)
        data_size_mb = len(npz_data) / (1024 * 1024)

        # Current slow method
        start = time.perf_counter()
        _ = simulate_pyodide_request_preparation(npz_data)
        slow_time = time.perf_counter() - start

        # Optimized method (memoryview is instant)
        start = time.perf_counter()
        _ = optimized_request_preparation_memoryview(npz_data)
        fast_time = time.perf_counter() - start

        speedup = slow_time / fast_time if fast_time > 0 else float("inf")

        print(
            f"\n[Request Preparation {size}x{size}] "
            f"Data: {data_size_mb:.2f} MB, "
            f"Slow: {slow_time * 1000:.1f} ms, "
            f"Fast: {fast_time * 1000:.1f} ms, "
            f"Speedup: {speedup:.1f}x"
        )

    def test_full_pipeline_comparison(self):
        """Compare full pipeline: serialize + pyodide overhead + deserialize."""
        size = 1024
        obj = create_test_image(size)
        data_size_mb = (size * size * 4) / (1024 * 1024)

        print(f"\n{'=' * 70}")
        print(
            f"Full Pipeline Comparison for {size}x{size} image ({data_size_mb:.2f} MB)"
        )
        print(f"{'=' * 70}")

        # Step 1: Serialize (compressed)
        start = time.perf_counter()
        npz_data = serialize_object_to_npz(obj, compress=True)
        serialize_time = time.perf_counter() - start

        compressed_mb = len(npz_data) / (1024 * 1024)
        print(
            f"Serialization (compressed): {serialize_time * 1000:.1f} ms "
            f"(size: {compressed_mb:.2f} MB)"
        )

        # Step 1b: Serialize (uncompressed - NEW!)
        start = time.perf_counter()
        npz_data_fast = serialize_object_to_npz(obj, compress=False)
        serialize_time_fast = time.perf_counter() - start

        uncompressed_mb = len(npz_data_fast) / (1024 * 1024)
        print(
            f"Serialization (uncompressed): {serialize_time_fast * 1000:.1f} ms "
            f"(size: {uncompressed_mb:.2f} MB)"
        )

        # Step 2a: Current Pyodide response conversion (SLOW)
        start = time.perf_counter()
        converted = simulate_pyodide_response_conversion(npz_data)
        slow_conversion_time = time.perf_counter() - start
        print(f"Response conversion (current): {slow_conversion_time * 1000:.1f} ms")

        # Step 2b: Optimized response conversion (FAST)
        start = time.perf_counter()
        optimized_response_conversion(npz_data)
        fast_conversion_time = time.perf_counter() - start
        print(f"Response conversion (optimized): {fast_conversion_time * 1000:.1f} ms")

        # Step 3: Deserialize
        start = time.perf_counter()
        deserialize_object_from_npz(converted)
        deserialize_time = time.perf_counter() - start
        print(f"Deserialization: {deserialize_time * 1000:.1f} ms")

        # Total comparison
        current_total = serialize_time + slow_conversion_time + deserialize_time
        optimized_total = serialize_time_fast + fast_conversion_time + deserialize_time

        print(f"\n{'=' * 70}")
        print(f"CURRENT TOTAL: {current_total * 1000:.1f} ms")
        print(f"OPTIMIZED TOTAL: {optimized_total * 1000:.1f} ms")
        print(f"IMPROVEMENT: {current_total / optimized_total:.1f}x faster")
        print(f"{'=' * 70}")

    @pytest.mark.parametrize("size", IMAGE_SIZES)
    def test_compression_vs_no_compression(self, size: int):
        """Compare serialization with and without compression."""
        obj = create_test_image(size)

        # Compressed
        start = time.perf_counter()
        npz_compressed = serialize_object_to_npz(obj, compress=True)
        compressed_time = time.perf_counter() - start
        compressed_size = len(npz_compressed)

        # Uncompressed
        start = time.perf_counter()
        npz_uncompressed = serialize_object_to_npz(obj, compress=False)
        uncompressed_time = time.perf_counter() - start
        uncompressed_size = len(npz_uncompressed)

        speedup = compressed_time / uncompressed_time if uncompressed_time > 0 else 1
        size_ratio = uncompressed_size / compressed_size if compressed_size > 0 else 1

        print(
            f"\n[Compression {size}x{size}] "
            f"Compressed: {compressed_time * 1000:.1f}ms "
            f"({compressed_size / 1024 / 1024:.2f}MB), "
            f"Uncompressed: {uncompressed_time * 1000:.1f}ms "
            f"({uncompressed_size / 1024 / 1024:.2f}MB), "
            f"Speed: {speedup:.1f}x, Size: {size_ratio:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
