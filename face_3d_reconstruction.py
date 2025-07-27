"""
3D Face Reconstruction System for Attendance
============================================

This module implements a complete 3D face reconstruction pipeline:
1. Capture multiple images with head pose variations
2. Extract face landmarks using MediaPipe
3. Estimate camera poses using Structure from Motion (SfM)
4. Triangulate 3D points to create a sparse point cloud
5. Build 3D mesh from point cloud
6. Compare 3D models for recognition

Author: AI Assistant
Date: 2025-01-23
"""

import cv2
import numpy as np
import mediapipe as mp
import open3d as o3d
from scipy.spatial.distance import cdist
from scipy.optimize import least_squares
from sklearn.neighbors import NearestNeighbors
import pickle
import os
import time
from typing import List, Tuple, Dict, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Face3DReconstructor:
    """Main class for 3D face reconstruction"""
    
    def __init__(self):
        """Initialize the 3D face reconstructor"""
        # MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Camera intrinsics (will be calibrated or estimated)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D face model template (468 landmarks from MediaPipe)
        self.face_model_3d = self._load_canonical_face_model()
        
        logger.info("Face3DReconstructor initialized")
    
    def _load_canonical_face_model(self) -> np.ndarray:
        """Load canonical 3D face model landmarks"""
        # MediaPipe's canonical face model (normalized coordinates)
        # These are the key facial landmarks in 3D space
        canonical_points = np.array([
            # Nose tip
            [0.0, 0.0, 0.0],
            # Left and right eye corners
            [-30.0, -125.0, -30.0], [30.0, -125.0, -30.0],
            # Mouth corners
            [-20.0, 170.0, -5.0], [20.0, 170.0, -5.0],
            # Chin
            [0.0, 200.0, -50.0],
            # Forehead
            [0.0, -180.0, -40.0],
            # Left and right cheeks
            [-100.0, 0.0, -80.0], [100.0, 0.0, -80.0],
        ])
        return canonical_points
    
    def calibrate_camera(self, images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate camera using face landmarks
        
        Args:
            images: List of calibration images
            
        Returns:
            camera_matrix, dist_coeffs
        """
        # For simplicity, we'll estimate camera parameters
        # In a real application, you'd use a calibration pattern
        height, width = images[0].shape[:2]
        
        # Estimate focal length (typical for webcams)
        focal_length = width * 0.8
        center = (width/2, height/2)
        
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))  # Assume no distortion
        
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        
        logger.info(f"Camera calibrated: focal_length={focal_length:.2f}")
        return camera_matrix, dist_coeffs
    
    def extract_face_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 2D face landmarks from image
        
        Args:
            image: Input image
            
        Returns:
            2D landmarks array or None if no face detected
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        face_landmarks = results.multi_face_landmarks[0]
        height, width = image.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        landmarks_2d = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            landmarks_2d.append([x, y])
        
        return np.array(landmarks_2d, dtype=np.float32)
    
    def estimate_head_pose(self, landmarks_2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate head pose using PnP algorithm
        
        Args:
            landmarks_2d: 2D face landmarks
            
        Returns:
            rotation_vector, translation_vector
        """
        if self.camera_matrix is None:
            raise ValueError("Camera not calibrated. Call calibrate_camera() first.")
        
        # Select key points for pose estimation
        key_indices = [1, 33, 61, 199, 263, 291]  # nose, eyes, mouth corners
        
        if len(landmarks_2d) < max(key_indices):
            # Use first few points if we don't have all landmarks
            key_indices = list(range(min(9, len(landmarks_2d))))
        
        object_pts = self.face_model_3d[:len(key_indices)]
        image_pts = landmarks_2d[key_indices]
        
        # Solve PnP problem
        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_pts, image_pts, self.camera_matrix, self.dist_coeffs
        )
        
        if not success:
            logger.warning("PnP solution failed")
            return np.zeros(3), np.zeros(3)
        
        return rotation_vector.flatten(), translation_vector.flatten()
    
    def triangulate_3d_points(self, 
                             landmarks_sequence: List[np.ndarray],
                             poses_sequence: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Triangulate 3D points from multiple views
        
        Args:
            landmarks_sequence: List of 2D landmarks for each frame
            poses_sequence: List of (rotation_vector, translation_vector) for each frame
            
        Returns:
            3D point cloud
        """
        if len(landmarks_sequence) < 2:
            raise ValueError("Need at least 2 views for triangulation")
        
        # Use first two views for triangulation
        landmarks1 = landmarks_sequence[0]
        landmarks2 = landmarks_sequence[1]
        
        rvec1, tvec1 = poses_sequence[0]
        rvec2, tvec2 = poses_sequence[1]
        
        # Convert rotation vectors to matrices
        R1, _ = cv2.Rodrigues(rvec1)
        R2, _ = cv2.Rodrigues(rvec2)
        
        # Create projection matrices
        P1 = self.camera_matrix @ np.hstack([R1, tvec1.reshape(-1, 1)])
        P2 = self.camera_matrix @ np.hstack([R2, tvec2.reshape(-1, 1)])
        
        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, landmarks1.T, landmarks2.T)
        
        # Convert from homogeneous to 3D coordinates
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def create_mesh_from_points(self, points_3d: np.ndarray) -> o3d.geometry.TriangleMesh:
        """
        Create 3D mesh from point cloud
        
        Args:
            points_3d: 3D point cloud
            
        Returns:
            Open3D triangle mesh
        """
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Create mesh using Poisson reconstruction
        try:
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=8, width=0, scale=1.1, linear_fit=False
            )
            
            # Clean up mesh
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Poisson reconstruction failed: {e}")
            # Fallback to convex hull
            hull, _ = pcd.compute_convex_hull()
            return hull
    
    def extract_face_features(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        """
        Extract geometric features from 3D face mesh (pickle-compatible)
        
        Args:
            mesh: 3D face mesh
            
        Returns:
            Dictionary of serializable features
        """
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) == 0:
            return {}
        
        # Get bounding box as min/max coordinates (serializable)
        bbox = mesh.get_axis_aligned_bounding_box()
        bbox_min = np.asarray(bbox.min_bound)
        bbox_max = np.asarray(bbox.max_bound)
        
        # Basic geometric features (all serializable)
        features = {
            'num_vertices': int(len(vertices)),
            'num_triangles': int(len(mesh.triangles)),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'bbox_size': (bbox_max - bbox_min).tolist(),
            'center': vertices.mean(axis=0).tolist(),
            'volume': float(mesh.get_volume() if mesh.is_watertight() else 0),
            'surface_area': float(mesh.get_surface_area()),
        }
        
        # Statistical features (convert to lists for serialization)
        features.update({
            'vertex_std': vertices.std(axis=0).tolist(),
            'vertex_range': (vertices.max(axis=0) - vertices.min(axis=0)).tolist(),
            'vertex_mean': vertices.mean(axis=0).tolist(),
        })
        
        # Mesh quality metrics
        try:
            edge_lengths = np.array(mesh.compute_edge_lengths())
            features['avg_edge_length'] = float(edge_lengths.mean())
            features['edge_length_std'] = float(edge_lengths.std())
            features['min_edge_length'] = float(edge_lengths.min())
            features['max_edge_length'] = float(edge_lengths.max())
        except:
            features['avg_edge_length'] = 0.0
            features['edge_length_std'] = 0.0
            features['min_edge_length'] = 0.0
            features['max_edge_length'] = 0.0
        
        return features
    
    def compare_3d_models(self, 
                         mesh1: o3d.geometry.TriangleMesh, 
                         mesh2: o3d.geometry.TriangleMesh,
                         method='icp') -> float:
        """
        Compare two 3D face models
        
        Args:
            mesh1: First 3D face mesh
            mesh2: Second 3D face mesh
            method: Comparison method ('icp', 'hausdorff', 'features')
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if method == 'icp':
            return self._compare_icp(mesh1, mesh2)
        elif method == 'hausdorff':
            return self._compare_hausdorff(mesh1, mesh2)
        elif method == 'features':
            return self._compare_features(mesh1, mesh2)
        else:
            raise ValueError(f"Unknown comparison method: {method}")
    
    def _compare_icp(self, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> float:
        """Compare meshes using Iterative Closest Point (ICP) algorithm"""
        try:
            pcd1 = mesh1.sample_points_uniformly(number_of_points=1000)
            pcd2 = mesh2.sample_points_uniformly(number_of_points=1000)
            
            # Perform ICP registration
            threshold = 0.05  # 5cm threshold
            result = o3d.pipelines.registration.registration_icp(
                pcd1, pcd2, threshold,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            
            # Convert fitness to similarity score
            similarity = result.fitness
            return min(1.0, similarity * 2)  # Scale to 0-1 range
            
        except Exception as e:
            logger.warning(f"ICP comparison failed: {e}")
            return 0.0
    
    def _compare_hausdorff(self, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> float:
        """Compare meshes using Hausdorff distance"""
        try:
            pcd1 = mesh1.sample_points_uniformly(number_of_points=500)
            pcd2 = mesh2.sample_points_uniformly(number_of_points=500)
            
            points1 = np.asarray(pcd1.points)
            points2 = np.asarray(pcd2.points)
            
            # Compute Hausdorff distance
            distances1 = cdist(points1, points2)
            distances2 = cdist(points2, points1)
            
            hausdorff_dist = max(
                np.max(np.min(distances1, axis=1)),
                np.max(np.min(distances2, axis=1))
            )
            
            # Convert distance to similarity (assuming max reasonable distance is 0.2)
            similarity = max(0, 1 - hausdorff_dist / 0.2)
            return similarity
            
        except Exception as e:
            logger.warning(f"Hausdorff comparison failed: {e}")
            return 0.0
    
    def _compare_features(self, mesh1: o3d.geometry.TriangleMesh, mesh2: o3d.geometry.TriangleMesh) -> float:
        """Compare meshes using geometric features"""
        try:
            features1 = self.extract_face_features(mesh1)
            features2 = self.extract_face_features(mesh2)
            
            if not features1 or not features2:
                return 0.0
            
            # Compare numerical features
            similarity_scores = []
            
            for key in ['volume', 'surface_area', 'avg_edge_length']:
                if key in features1 and key in features2:
                    val1, val2 = features1[key], features2[key]
                    if val1 > 0 and val2 > 0:
                        ratio = min(val1, val2) / max(val1, val2)
                        similarity_scores.append(ratio)
            
            # Compare centers
            if 'center' in features1 and 'center' in features2:
                center_dist = np.linalg.norm(features1['center'] - features2['center'])
                center_sim = max(0, 1 - center_dist / 0.1)  # 10cm max distance
                similarity_scores.append(center_sim)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Feature comparison failed: {e}")
            return 0.0


class Face3DTrainingSession:
    """Handles 3D face training session"""
    
    def __init__(self, reconstructor: Face3DReconstructor, min_frame_interval=1.0):
        self.reconstructor = reconstructor
        self.captured_frames = []
        self.landmarks_sequence = []
        self.poses_sequence = []
        self.is_calibrated = False
        self.min_frame_interval = min_frame_interval  # Minimum time between frames in seconds
        self.last_capture_time = 0
        self.pose_variation_threshold = 10.0  # Minimum pose difference in degrees
    
    def add_frame(self, image: np.ndarray) -> bool:
        """
        Add a frame to the training session with timing and pose variation controls
        
        Args:
            image: Input image frame
            
        Returns:
            True if frame was successfully processed
        """
        current_time = time.time()
        
        # Check if enough time has passed since last capture
        if current_time - self.last_capture_time < self.min_frame_interval:
            return False
        
        # Extract landmarks
        landmarks = self.reconstructor.extract_face_landmarks(image)
        if landmarks is None:
            logger.warning("No face detected in frame")
            return False
        
        # Calibrate camera if not done yet
        if not self.is_calibrated:
            self.reconstructor.calibrate_camera([image])
            self.is_calibrated = True
        
        # Estimate pose
        try:
            rvec, tvec = self.reconstructor.estimate_head_pose(landmarks)
            
            # Check for sufficient pose variation if we already have frames
            if len(self.poses_sequence) > 0:
                last_rvec, _ = self.poses_sequence[-1]
                pose_diff = np.linalg.norm(rvec - last_rvec) * 180 / np.pi  # Convert to degrees
                
                if pose_diff < self.pose_variation_threshold:
                    logger.info(f"Insufficient pose variation ({pose_diff:.2f}°), skipping frame")
                    return False
            
            self.captured_frames.append(image.copy())
            self.landmarks_sequence.append(landmarks)
            self.poses_sequence.append((rvec, tvec))
            self.last_capture_time = current_time
            
            logger.info(f"Frame {len(self.captured_frames)} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process frame: {e}")
            return False
    
    def generate_3d_model(self) -> Optional[o3d.geometry.TriangleMesh]:
        """
        Generate 3D model from captured frames
        
        Returns:
            3D face mesh or None if failed
        """
        if len(self.landmarks_sequence) < 2:
            logger.error("Need at least 2 frames for 3D reconstruction")
            return None
        
        try:
            # Triangulate 3D points
            points_3d = self.reconstructor.triangulate_3d_points(
                self.landmarks_sequence, self.poses_sequence
            )
            
            # Create mesh
            mesh = self.reconstructor.create_mesh_from_points(points_3d)
            
            logger.info(f"3D model generated with {len(mesh.vertices)} vertices")
            return mesh
            
        except Exception as e:
            logger.error(f"Failed to generate 3D model: {e}")
            return None
    
    def save_session(self, filepath: str):
        """Save training session data"""
        session_data = {
            'landmarks_sequence': self.landmarks_sequence,
            'poses_sequence': self.poses_sequence,
            'camera_matrix': self.reconstructor.camera_matrix,
            'dist_coeffs': self.reconstructor.dist_coeffs,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(session_data, f)
        
        logger.info(f"Session saved to {filepath}")
    
    def load_session(self, filepath: str):
        """Load training session data"""
        with open(filepath, 'rb') as f:
            session_data = pickle.load(f)
        
        self.landmarks_sequence = session_data['landmarks_sequence']
        self.poses_sequence = session_data['poses_sequence']
        self.reconstructor.camera_matrix = session_data['camera_matrix']
        self.reconstructor.dist_coeffs = session_data['dist_coeffs']
        self.is_calibrated = True
        
        logger.info(f"Session loaded from {filepath}")


# Utility functions
def save_3d_model(mesh: o3d.geometry.TriangleMesh, filepath: str):
    """Save 3D mesh to file"""
    success = o3d.io.write_triangle_mesh(filepath, mesh)
    if success:
        logger.info(f"3D model saved to {filepath}")
    else:
        logger.error(f"Failed to save 3D model to {filepath}")
    return success

def load_3d_model(filepath: str) -> Optional[o3d.geometry.TriangleMesh]:
    """Load 3D mesh from file"""
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    mesh = o3d.io.read_triangle_mesh(filepath)
    if len(mesh.vertices) == 0:
        logger.error(f"Failed to load 3D model from {filepath}")
        return None
    
    logger.info(f"3D model loaded from {filepath}")
    return mesh

def visualize_3d_model(mesh: o3d.geometry.TriangleMesh, title: str = "3D Face Model"):
    """Visualize 3D mesh using Open3D"""
    try:
        # Add colors to mesh
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        mesh.compute_vertex_normals()
        
        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)
        vis.add_geometry(mesh)
        vis.run()
        vis.destroy_window()
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")

if __name__ == "__main__":
    # Test the 3D reconstruction system
    reconstructor = Face3DReconstructor()
    session = Face3DTrainingSession(reconstructor)
    
    # Test with webcam (if available)
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("Press 'c' to capture frame, 'q' to quit, 'g' to generate 3D model")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                cv2.imshow("3D Face Training", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    success = session.add_frame(frame)
                    if success:
                        print(f"Frame captured! Total frames: {len(session.captured_frames)}")
                elif key == ord('g'):
                    mesh = session.generate_3d_model()
                    if mesh:
                        visualize_3d_model(mesh)
                elif key == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Camera test failed: {e}")
