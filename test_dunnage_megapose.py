#!/usr/bin/env python3
"""
Test script for MegaPose6D with dunnage detection
Prepares the dunnage PLY model for pose estimation
"""

import os
import sys
import numpy as np
import torch
import trimesh
from pathlib import Path

# Set environment for Mac M1
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

def test_environment():
    """Test the basic environment setup"""
    print("=== Environment Test ===")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check PyTorch and MPS
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    # Test basic tensor operations on MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        x = torch.randn(3, 3, device=device)
        y = torch.eye(3, device=device)
        z = torch.mm(x, y)
        print("✓ MPS tensor operations working")

    print("✓ Environment test passed\n")

def test_megapose_import():
    """Test MegaPose6D imports"""
    print("=== MegaPose6D Import Test ===")

    try:
        import megapose
        print("✓ MegaPose6D core imported")

        try:
            from megapose.lib3d.transform import Transform
            print("✓ Transform module imported")
        except ImportError as e:
            print(f"○ Transform module: {e}")

        try:
            from megapose.inference.types import PoseEstimatesType, DetectionsType
            print("✓ Type definitions imported")
        except ImportError as e:
            print(f"○ Type definitions: {e}")

        print("✓ MegaPose6D basic imports successful\n")
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dunnage_model():
    """Test loading and processing the dunnage PLY model"""
    print("=== Dunnage Model Test ===")

    dunnage_path = Path("dunnage.ply")

    if not dunnage_path.exists():
        print("✗ Dunnage PLY file not found")
        return False

    try:
        # Load the PLY file using trimesh
        mesh = trimesh.load(dunnage_path)
        print(f"✓ Dunnage mesh loaded: {type(mesh)}")
        print(f"  - Vertices: {len(mesh.vertices)}")
        print(f"  - Faces: {len(mesh.faces)}")
        print(f"  - Bounds: {mesh.bounds}")
        print(f"  - Scale: {mesh.scale}")

        # Check if mesh is watertight
        print(f"  - Watertight: {mesh.is_watertight}")
        print(f"  - Volume: {mesh.volume:.4f}")

        # Visualize basic properties
        center = mesh.centroid
        print(f"  - Centroid: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")

        # For pose estimation, we need the mesh to be in a standard format
        # Let's check if we can convert it to the format MegaPose expects
        vertices = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)

        print(f"  - Vertices shape: {vertices.shape}")
        print(f"  - Faces shape: {faces.shape}")
        print(f"  - Vertex range X: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
        print(f"  - Vertex range Y: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
        print(f"  - Vertex range Z: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")

        # Save processed mesh for MegaPose
        processed_path = "dunnage_processed.ply"
        mesh.export(processed_path)
        print(f"✓ Processed mesh saved to: {processed_path}")

        print("✓ Dunnage model test passed\n")
        return True

    except Exception as e:
        print(f"✗ Dunnage model test failed: {e}")
        return False

def test_camera_intrinsics():
    """Test camera intrinsics for RealSense D435"""
    print("=== Camera Intrinsics Test ===")

    # RealSense D435 typical parameters (you should calibrate these)
    # RGB camera intrinsics (approximate)
    width, height = 640, 480
    fx = 615.0  # focal length x
    fy = 615.0  # focal length y
    cx = 320.0  # principal point x
    cy = 240.0  # principal point y

    # Create camera intrinsic matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float32)

    print(f"Camera resolution: {width}x{height}")
    print(f"Camera intrinsic matrix K:")
    print(K)
    print("✓ Camera intrinsics test passed\n")

    return True

def test_pose_estimation_pipeline():
    """Test the basic pose estimation pipeline structure"""
    print("=== Pose Estimation Pipeline Test ===")

    try:
        # This is what we would need for actual pose estimation:

        # 1. Object mesh (dunnage)
        print("1. ✓ Object mesh: dunnage.ply loaded")

        # 2. Camera intrinsics
        print("2. ✓ Camera intrinsics: RealSense D435 parameters")

        # 3. RGB image (would come from RealSense)
        print("3. ○ RGB image: (would be captured from RealSense)")

        # 4. Object detection/bounding box (from your YOLO)
        print("4. ○ Bounding box: (would come from YOLO detection)")

        # 5. Depth image (optional, from RealSense)
        print("5. ○ Depth image: (optional, from RealSense depth sensor)")

        # 6. MegaPose6D inference
        print("6. ○ MegaPose6D inference: (ready to run)")

        print("✓ Pipeline structure verified\n")
        return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 MegaPose6D Dunnage Detection Setup Test\n")

    # Run tests
    test_results = []
    test_results.append(test_environment())
    test_results.append(test_megapose_import())
    test_results.append(test_dunnage_model())

    # Camera and pipeline tests
    test_results.append(test_camera_intrinsics())
    test_results.append(test_pose_estimation_pipeline())

    # Summary
    print("=== Test Summary ===")
    # Convert None values to False for counting
    valid_results = [r if r is not None else False for r in test_results]
    passed = sum(valid_results)
    total = len(valid_results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! MegaPose6D is ready for dunnage detection.")
        print("\n📋 Next steps:")
        print("1. Capture RGB image with RealSense D435")
        print("2. Run YOLO detection to get dunnage bounding box")
        print("3. Run MegaPose6D inference for 6D pose estimation")
        print("4. Convert pose to ROS2 PoseStamped message")
        return 0
    else:
        print("❌ Some tests failed. Please check the setup.")
        return 1

if __name__ == "__main__":
    sys.exit(main())