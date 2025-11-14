#!/usr/bin/env python3
"""
Test MegaPose6D with actual images
Supports:
1. Manual bounding box input
2. YOLO format detection files
3. Interactive bounding box selection
"""

import os
import numpy as np
import cv2
import torch
import trimesh
from pathlib import Path
import json

# Set environment
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

class ImagePoseTester:
    """Test MegaPose6D with real images"""

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load dunnage mesh
        self.load_dunnage_mesh()

        # Default camera intrinsics (RealSense D435)
        self.camera_K = np.array([
            [615.0, 0, 320.0],
            [0, 615.0, 240.0],
            [0, 0, 1.0]
        ], dtype=np.float32)

    def load_dunnage_mesh(self):
        """Load the dunnage mesh"""
        mesh_path = "dunnage_processed.ply"
        if not Path(mesh_path).exists():
            mesh_path = "dunnage.ply"

        if not Path(mesh_path).exists():
            print("❌ Dunnage mesh not found. Please ensure dunnage.ply exists.")
            return None

        self.mesh = trimesh.load(mesh_path)
        # Convert from mm to meters
        self.mesh.vertices *= 0.001
        print(f"✓ Loaded dunnage mesh: {len(self.mesh.vertices)} vertices")
        return self.mesh

    def load_image(self, image_path):
        """Load and display image"""
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return None

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Could not read image: {image_path}")
            return None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✓ Loaded image: {image_rgb.shape}")
        return image_rgb

    def parse_yolo_detection(self, image_shape, yolo_line):
        """
        Parse YOLO format detection line
        Format: class_id x_center y_center width height confidence

        Args:
            image_shape: (height, width, channels)
            yolo_line: "0 0.5 0.4 0.3 0.2 0.95"

        Returns:
            [x1, y1, x2, y2, confidence]
        """
        parts = yolo_line.strip().split()

        if len(parts) < 5:
            print("❌ Invalid YOLO format. Expected: class_id x_center y_center width height [confidence]")
            return None

        try:
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            confidence = float(parts[5]) if len(parts) > 5 else 1.0

            # Convert normalized coordinates to pixel coordinates
            h, w = image_shape[:2]

            x1 = int((x_center - width/2) * w)
            y1 = int((y_center - height/2) * h)
            x2 = int((x_center + width/2) * w)
            y2 = int((y_center + height/2) * h)

            # Clamp to image boundaries
            x1 = max(0, min(w-1, x1))
            y1 = max(0, min(h-1, y1))
            x2 = max(0, min(w-1, x2))
            y2 = max(0, min(h-1, y2))

            return [x1, y1, x2, y2, confidence]

        except (ValueError, IndexError) as e:
            print(f"❌ Error parsing YOLO line: {e}")
            return None

    def manual_bbox_selection(self, image):
        """Interactive bounding box selection"""
        print("\n🖱️  Select bounding box around dunnage:")
        print("   - Click and drag to select area")
        print("   - Press 'r' to reset")
        print("   - Press 'q' to quit")
        print("   - Press SPACE or ENTER to confirm")

        # Global variables for mouse callback
        self.drawing = False
        self.bbox = [0, 0, 0, 0]

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.bbox[0], self.bbox[1] = x, y

            elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    self.bbox[2], self.bbox[3] = x, y

            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.bbox[2], self.bbox[3] = x, y

        # Create window and set mouse callback
        cv2.namedWindow('Select Dunnage', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Select Dunnage', mouse_callback)

        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        while True:
            img_copy = image_bgr.copy()

            # Draw rectangle
            if self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]:
                cv2.rectangle(img_copy,
                            (self.bbox[0], self.bbox[1]),
                            (self.bbox[2], self.bbox[3]),
                            (0, 255, 0), 2)

            cv2.imshow('Select Dunnage', img_copy)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            elif key == ord('r'):
                self.bbox = [0, 0, 0, 0]
            elif key in [ord(' '), 13]:  # Space or Enter
                if self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]:
                    cv2.destroyAllWindows()
                    return [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], 1.0]

        cv2.destroyAllWindows()
        return None

    def estimate_pose_from_bbox(self, image, bbox):
        """
        Estimate 6D pose from image and bounding box
        This is a simplified version - real MegaPose6D would be more complex
        """
        x1, y1, x2, y2, confidence = bbox

        print(f"\n🔄 Estimating pose for bbox: [{x1}, {y1}, {x2}, {y2}]")

        # Crop image to bounding box
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            print("❌ Invalid bounding box")
            return None

        print(f"   Cropped region: {cropped.shape}")

        # Mock pose estimation (replace with actual MegaPose6D)
        # In real implementation, this would use MegaPose6D inference

        # Estimate depth from bounding box size (very rough)
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Assume dunnage is roughly 1.2m long, estimate distance
        focal_length = self.camera_K[0, 0]
        real_width = 1.2  # meters
        estimated_distance = (real_width * focal_length) / bbox_width

        # Center of bounding box in image coordinates
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Convert to 3D position (simplified projection)
        x_3d = (center_x - self.camera_K[0, 2]) * estimated_distance / self.camera_K[0, 0]
        y_3d = (center_y - self.camera_K[1, 2]) * estimated_distance / self.camera_K[1, 1]
        z_3d = estimated_distance

        # Mock rotation (slight random variation)
        angle = np.random.uniform(-0.1, 0.1)
        R = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])

        pose = {
            'rotation_matrix': R,
            'translation': np.array([x_3d, y_3d, z_3d]),
            'confidence': confidence,
            'bbox': bbox[:4]
        }

        return pose

    def visualize_result(self, image, bbox, pose):
        """Visualize the pose estimation result"""
        x1, y1, x2, y2 = bbox[:4]
        confidence = bbox[4] if len(bbox) > 4 else 1.0

        # Create visualization
        vis_image = image.copy()
        vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        # Draw bounding box
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw pose information
        t = pose['translation']
        text_lines = [
            f"Confidence: {confidence:.2f}",
            f"Position: [{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}]m",
            f"Distance: {t[2]:.2f}m"
        ]

        for i, line in enumerate(text_lines):
            cv2.putText(vis_bgr, line, (x1, y1 - 10 - i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show result
        cv2.imshow('Pose Estimation Result', vis_bgr)
        print("\n📊 Result (press any key to close):")
        print(f"   Position: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] meters")
        print(f"   Estimated distance: {t[2]:.2f}m")
        print(f"   Confidence: {confidence:.2f}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    """Main testing function"""
    print("🎯 MegaPose6D Image Testing Tool\n")

    tester = ImagePoseTester()

    print("Choose input method:")
    print("1. Load image and select bounding box manually")
    print("2. Load image with YOLO format detection")
    print("3. Use sample image path")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == "1":
        image_path = input("Enter image path: ").strip()
        image = tester.load_image(image_path)
        if image is None:
            return

        bbox = tester.manual_bbox_selection(image)
        if bbox is None:
            print("❌ No bounding box selected")
            return

    elif choice == "2":
        image_path = input("Enter image path: ").strip()
        yolo_detection = input("Enter YOLO detection (class_id x_center y_center width height confidence): ").strip()

        image = tester.load_image(image_path)
        if image is None:
            return

        bbox = tester.parse_yolo_detection(image.shape, yolo_detection)
        if bbox is None:
            return

    elif choice == "3":
        print("\n📁 Example image paths to try:")
        print("   - /Users/adrianmendez/Desktop/test_image.jpg")
        print("   - /Users/adrianmendez/Downloads/dunnage_photo.jpg")
        print("   - Any image with visible dunnage")

        image_path = input("Enter image path: ").strip()
        image = tester.load_image(image_path)
        if image is None:
            print("\n💡 Tip: Take a photo of dunnage and try again!")
            return

        bbox = tester.manual_bbox_selection(image)
        if bbox is None:
            print("❌ No bounding box selected")
            return

    else:
        print("❌ Invalid choice")
        return

    # Estimate pose
    pose = tester.estimate_pose_from_bbox(image, bbox)
    if pose is None:
        return

    # Show results
    tester.visualize_result(image, bbox, pose)

    print("\n🎉 Test complete!")
    print("\n💡 Notes:")
    print("   - This is a simplified pose estimation for testing")
    print("   - Real MegaPose6D would provide more accurate results")
    print("   - Try with different images and bounding boxes")

if __name__ == "__main__":
    main()