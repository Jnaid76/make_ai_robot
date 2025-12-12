#!/usr/bin/env python3
"""
Unit tests for map_raycast module.

Tests:
1. Map loading and coordinate conversion
2. Bresenham raycast accuracy
3. Likelihood computation
"""

import sys
import os
import numpy as np
import math

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(__file__))

from map_raycast import OccupancyGridMap, RaycastSensor, LikelihoodFieldModel, create_map_raycast_model


def test_map_loading():
    """Test map loading and coordinate conversion."""
    print("\n=== Test 1: Map Loading ===")
    
    map_path = "src/go1_simulation/maps/hospital.yaml"
    if not os.path.exists(map_path):
        print(f"SKIP: Map file not found at {map_path}")
        return False
    
    occ_map = OccupancyGridMap(map_path)
    
    # Check dimensions
    assert occ_map.width == 1000, f"Expected width 1000, got {occ_map.width}"
    assert occ_map.height == 1000, f"Expected height 1000, got {occ_map.height}"
    assert occ_map.resolution == 0.1, f"Expected resolution 0.1, got {occ_map.resolution}"
    
    # Check coordinate conversion
    # World origin should map to grid (0, 0)
    row, col = occ_map.world_to_grid(-50.0, -50.0)
    assert row == 0 and col == 0, f"Origin conversion failed: {row}, {col}"
    
    # Center of map
    row, col = occ_map.world_to_grid(0.0, 0.0)
    assert row == 500 and col == 500, f"Center conversion failed: {row}, {col}"
    
    # Reverse conversion
    x, y = occ_map.grid_to_world(500, 500)
    assert abs(x - 0.0) < 0.01 and abs(y - 0.0) < 0.01, f"Reverse conversion failed: {x}, {y}"
    
    print("✓ Map loading and coordinate conversion correct")
    print(f"  Map size: {occ_map.width}x{occ_map.height}, resolution: {occ_map.resolution}")
    print(f"  Occupied cells: {np.sum(occ_map.grid == 100)}")
    
    return True


def test_raycast_accuracy():
    """Test raycast in open space and against walls."""
    print("\n=== Test 2: Raycast Accuracy ===")
    
    map_path = "src/go1_simulation/maps/hospital.yaml"
    if not os.path.exists(map_path):
        print(f"SKIP: Map file not found")
        return False
    
    occ_map = OccupancyGridMap(map_path)
    raycast = RaycastSensor(occ_map, max_range=30.0)
    
    # Use a known free space position (from map inspection)
    # Center (0,0) is occupied, so use (2.0, 7.5) which is free
    x0, y0 = 2.0, 7.5
    
    # Test 1: Rays in different directions
    angles = [0, math.pi/2, math.pi, -math.pi/2]  # E, N, W, S
    
    ranges = []
    for angle in angles:
        r = raycast.bresenham_raycast(x0, y0, angle)
        ranges.append(r)
        direction = ['E', 'N', 'W', 'S'][len(ranges)-1]
        print(f"  Ray from ({x0}, {y0}) direction {direction}: {r:.2f}m")
    
    # All rays should return valid positive ranges
    assert all(r > 0 for r in ranges), f"Some ranges are zero or negative: {ranges}"
    
    # At least some rays should hit something (not all at max_range)
    # but in open hospital, some might reach max_range
    print(f"  Rays at max_range: {sum(1 for r in ranges if r >= 29.9)} / {len(ranges)}")
    
    # Test 2: Cast multiple rays (full scan)
    scan_angles = np.linspace(-math.pi, math.pi, 360)
    expected_ranges = raycast.cast_rays(np.array([x0, y0, 0.0]), scan_angles)
    
    print(f"  Full 360° scan from ({x0}, {y0}):")
    print(f"    Mean range: {np.mean(expected_ranges):.2f}m")
    print(f"    Min range: {np.min(expected_ranges):.2f}m")
    print(f"    Max range: {np.max(expected_ranges):.2f}m")
    print(f"    Rays at max_range: {np.sum(expected_ranges >= 29.9)} / {len(expected_ranges)}")
    
    # Sanity checks
    assert np.all(expected_ranges > 0), "Some ranges are zero or negative"
    assert np.all(expected_ranges <= 30.0), "Some ranges exceed max_range"
    
    print("✓ Raycast accuracy test passed")
    
    return True


def test_likelihood_computation():
    """Test likelihood field measurement model."""
    print("\n=== Test 3: Likelihood Computation ===")
    
    likelihood_model = LikelihoodFieldModel(sigma_hit=0.2, max_range=30.0)
    
    # Test 1: Perfect match should give high likelihood
    actual = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    likelihood_perfect = likelihood_model.compute_likelihood(actual, expected)
    print(f"  Perfect match likelihood: {likelihood_perfect:.6f}")
    
    # Test 2: Small error should give good likelihood
    expected_small_error = np.array([1.05, 2.05, 3.05, 4.05, 5.05])
    likelihood_small_error = likelihood_model.compute_likelihood(actual, expected_small_error)
    print(f"  Small error (5cm) likelihood: {likelihood_small_error:.6f}")
    
    # Test 3: Large error should give low likelihood
    expected_large_error = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    likelihood_large_error = likelihood_model.compute_likelihood(actual, expected_large_error)
    print(f"  Large error (1m) likelihood: {likelihood_large_error:.6f}")
    
    # Verify ordering
    assert likelihood_perfect > likelihood_small_error, "Perfect should be better than small error"
    assert likelihood_small_error > likelihood_large_error, "Small error should be better than large error"
    
    # Test 4: Multiple particles weight computation
    expected_list = [
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),  # Perfect
        np.array([1.1, 2.1, 3.1, 4.1, 5.1]),  # Good
        np.array([2.0, 3.0, 4.0, 5.0, 6.0]),  # Bad
    ]
    
    weights = likelihood_model.compute_weights(actual, expected_list)
    print(f"  Weights for 3 particles: {weights}")
    
    # Verify properties
    assert abs(np.sum(weights) - 1.0) < 1e-6, "Weights should sum to 1"
    assert weights[0] > weights[1] > weights[2], "Weights should be ordered by accuracy"
    
    print("✓ Likelihood computation test passed")
    
    return True


def test_full_pipeline():
    """Test full pipeline: map + raycast + likelihood."""
    print("\n=== Test 4: Full Pipeline ===")
    
    map_path = "src/go1_simulation/maps/hospital.yaml"
    if not os.path.exists(map_path):
        print(f"SKIP: Map file not found")
        return False
    
    # Create all components
    occ_map, raycast_sensor, likelihood_model = create_map_raycast_model(
        map_path,
        max_range=30.0,
        sigma_hit=0.2,
        n_beams=144
    )
    
    # Use free space position
    x0, y0 = 2.0, 7.5
    
    # Simulate particles at different positions (all in free space)
    particles = np.array([
        [x0, y0, 0.0],           # Reference position, facing east
        [x0+1.0, y0, 0.0],       # 1m east
        [x0, y0+1.0, 0.0],       # 1m north
        [x0+2.0, y0+2.0, 0.0],   # 2m northeast
    ])
    
    # Simulate actual scan from first particle position
    scan_angles = np.linspace(-math.pi, math.pi, 144)
    actual_scan = raycast_sensor.cast_rays(particles[0], scan_angles)
    
    print(f"  Actual scan from particle 0: mean={np.mean(actual_scan):.2f}m, min={np.min(actual_scan):.2f}m")
    
    # Verify scan is valid
    assert np.all(actual_scan > 0), "Actual scan has zero/negative ranges"
    
    # Compute expected ranges for all particles
    expected_list = []
    for i, particle in enumerate(particles):
        expected = raycast_sensor.cast_rays(particle, scan_angles)
        expected_list.append(expected)
        print(f"  Particle {i} at ({particle[0]:.1f}, {particle[1]:.1f}): mean range={np.mean(expected):.2f}m")
    
    # Compute weights
    weights = likelihood_model.compute_weights(actual_scan, expected_list)
    print(f"  Weights: {weights}")
    print(f"  Best particle: {np.argmax(weights)} with weight {np.max(weights):.4f}")
    
    # Particle 0 should have highest weight (perfect match)
    assert np.argmax(weights) == 0, "Particle 0 should have highest weight"
    assert weights[0] > 0.3, "Best particle should have significantly higher weight than others"
    
    print("✓ Full pipeline test passed")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Map Raycast Module - Unit Tests")
    print("=" * 60)
    
    tests = [
        test_map_loading,
        test_raycast_accuracy,
        test_likelihood_computation,
        test_full_pipeline,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
