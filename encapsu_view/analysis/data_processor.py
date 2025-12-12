from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import cv2

from encapsu_view.entities.capsule import Capsule
from encapsu_view.entities.contour import Contour
from encapsu_view.entities.ellipse import Ellipse
from encapsu_view.entities.processed_capsule import ProcessedCapsule


class DataProcessor:

    @staticmethod
    def visualize_contour_filtering(contour: 'Contour', 
                                  line_filtration_method: Optional[str] = None,
                                  line_filtration_params: Optional[Dict[str, Any]] = None,
                                  save_path: Optional[str] = None) -> None:
        interpolate_contour = contour.interpolate_to_max_distance()
        points = interpolate_contour.points.astype(np.float32)
        
        if line_filtration_method is not None:
            filtered_points = DataProcessor._filter_contour_points(
                points, line_filtration_method, line_filtration_params or {}
            )
        else:
            filtered_points = points
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        all_points = np.vstack([points, filtered_points])
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 10)
        
        ax1.scatter(points[:, 0], points[:, 1], c='blue', s=20, alpha=0.7, label='Original points')
        ax1.plot(points[:, 0], points[:, 1], 'b-', alpha=0.5, linewidth=1)
        ax1.set_title('Original contour\n(Total points: {})'.format(len(points)))
        ax1.set_xlabel('X coord')
        ax1.set_ylabel('Y coord')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.set_xlim(x_min - margin, x_max + margin)
        ax1.set_ylim(y_min - margin, y_max + margin)
        
        colors = ['green' if len(filtered_points) >= 5 else 'red']
        label = 'Filtered points ({})'.format(len(filtered_points))
        
        ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], c=colors[0], s=20, alpha=0.7, label=label)
        
        if len(filtered_points) > 1:
            ax2.plot(filtered_points[:, 0], filtered_points[:, 1], 'g-', alpha=0.5, linewidth=1)
        
        removed_points = DataProcessor._get_removed_points(points, filtered_points)
        if len(removed_points) > 0:
            ax2.scatter(removed_points[:, 0], removed_points[:, 1], c='red', s=15, alpha=0.5, 
                       label='Removed points ({})'.format(len(removed_points)))
        
        ax2.set_title('Contour after filtering\Method: {}'.format(
            line_filtration_method if line_filtration_method else 'Without filtering'))
        ax2.set_xlabel('X coord')
        ax2.set_ylabel('Y coord')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.set_xlim(x_min - margin, x_max + margin)
        ax2.set_ylim(y_min - margin, y_max + margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Image saved: {save_path}")
        
        plt.show()
        

        print("\n=== FILTERING STATISTICS ===")
        print(f"Original points: {len(points)}")
        print(f"Remaining after filtering: {len(filtered_points)}")
        print(f"Removed points: {len(removed_points)}")
        print(f"Percentage of points saved: {len(filtered_points)/len(points)*100:.1f}%")
        
        if len(filtered_points) < 5:
            print("ATTENTION: Not enough points to approximate the ellipse!")
    
    @staticmethod
    def _get_removed_points(original_points: np.ndarray, filtered_points: np.ndarray) -> np.ndarray:

        if len(filtered_points) == 0:
            return original_points
        

        removed_indices = []
        
        for i, point in enumerate(original_points):
            distances = np.sqrt(np.sum((filtered_points - point) ** 2, axis=1))
            min_distance = np.min(distances)
            

            if min_distance > 1.0: 
                removed_indices.append(i)
        
        return original_points[removed_indices]
    
    @staticmethod
    def compare_ellipse_fitting(contour: 'Contour',
                              line_filtration_method: Optional[str] = None,
                              line_filtration_params: Optional[Dict[str, Any]] = None,
                              save_path: Optional[str] = None) -> None:

        interpolate_contour = contour.interpolate_to_max_distance()
        points = interpolate_contour.points.astype(np.float32)
        

        try:
            (x1, y1), (ma1, mi1), angle1 = cv2.fitEllipse(points)
            ellipse1 = Ellipse(center=(x1, y1), axes=(ma1, mi1), angle=angle1)
        except Exception as e:
            print(f"Error during approximation without filtering: {e}")
            ellipse1 = None
        
        if line_filtration_method is not None:
            filtered_points = DataProcessor._filter_contour_points(
                points, line_filtration_method, line_filtration_params or {}
            )
            if len(filtered_points) >= 5:
                try:
                    (x2, y2), (ma2, mi2), angle2 = cv2.fitEllipse(filtered_points)
                    ellipse2 = Ellipse(center=(x2, y2), axes=(ma2, mi2), angle=angle2)
                except Exception as e:
                    print(f"Error during approximation with filtering: {e}")
                    ellipse2 = None
            else:
                ellipse2 = None
                print("Not enough points for approximation after filtering")
        else:
            ellipse2 = None
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        all_points = points if ellipse2 is None else np.vstack([points, filtered_points])
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        margin = max((x_max - x_min) * 0.1, (y_max - y_min) * 0.1, 10)
        
        ax1.scatter(points[:, 0], points[:, 1], c='blue', s=15, alpha=0.6)
        if ellipse1 is not None:
            DataProcessor._plot_ellipse(ax1, ellipse1, 'red', 'Without filtering')
        ax1.set_title('Approximation without filtering')
        ax1.set_xlabel('X coordinate')
        ax1.set_ylabel('Y coordinate')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.set_xlim(x_min - margin, x_max + margin)
        ax1.set_ylim(y_min - margin, y_max + margin)
        
        if line_filtration_method is not None:
            ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], c='green', s=15, alpha=0.6, 
                       label='Filtered points')
            removed_points = DataProcessor._get_removed_points(points, filtered_points)
            if len(removed_points) > 0:
                ax2.scatter(removed_points[:, 0], removed_points[:, 1], c='red', s=10, alpha=0.4, 
                           label='Removed points')
        else:
            ax2.scatter(points[:, 0], points[:, 1], c='blue', s=15, alpha=0.6)
        
        if ellipse2 is not None:
            DataProcessor._plot_ellipse(ax2, ellipse2, 'purple', 'With filtering')
        
        ax2.set_title('Approximation with filtering\n({})'.format(line_filtration_method))
        ax2.set_xlabel('X coordinate')
        ax2.set_ylabel('Y coordinate')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_aspect('equal')
        ax2.set_xlim(x_min - margin, x_max + margin)
        ax2.set_ylim(y_min - margin, y_max + margin)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        

        if ellipse1 is not None and ellipse2 is not None:
            print("\n=== COMPARATIVE STATISTICS OF ELLIPSES ===")
            print(f"Without filtering: center=({ellipse1.center[0]:.1f}, {ellipse1.center[1]:.1f}), "
                  f"axes=({ellipse1.axes[0]:.1f}, {ellipse1.axes[1]:.1f}), angle={ellipse1.angle:.1f}°")
            print(f"With filtering:  center=({ellipse2.center[0]:.1f}, {ellipse2.center[1]:.1f}), "
                  f"axes=({ellipse2.axes[0]:.1f}, {ellipse2.axes[1]:.1f}), angle={ellipse2.angle:.1f}°")
    
    @staticmethod
    def _plot_ellipse(ax, ellipse: 'Ellipse', color: str, label: str):
        """Допоміжна функція для візуалізації еліпса"""
        from matplotlib.patches import Ellipse as MplEllipse
        
        patch = MplEllipse(ellipse.center, ellipse.axes[0], ellipse.axes[1], 
                          angle=ellipse.angle, fill=False, color=color, linewidth=2, label=label)
        ax.add_patch(patch)
        
        # Додаємо центр
        ax.plot(ellipse.center[0], ellipse.center[1], 'o', color=color, markersize=5)

    
    @staticmethod
    def fit_ellipse_to_contour(contour: Contour, method: str = "direct", 
                               line_filtration_method: Optional[str] = None, line_filtration_params: Optional[Dict[str, Any]] = None) -> Optional[Ellipse]:
        if len(contour.points) < 5:
            return None

        interpolate_contour = contour.interpolate_to_max_distance()
        points = interpolate_contour.points.astype(np.float32)

        if line_filtration_method is not None:
            filtered_points = DataProcessor._filter_contour_points(
                points, line_filtration_method, line_filtration_params or {}
            )
            if len(filtered_points) >= 5:
                points = filtered_points
            else:
                print("Warning: Too few points after line filtration, using original points")

        
        if method == "direct":
            try:
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(points)
                # (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(contour.points.astype(np.float32))
                return Ellipse(center=(x, y), axes=(major_axis, minor_axis), angle=angle)
            except Exception as e:
                print(f"Error during fitEllipse: {e}")
                return None
        
        elif method == "ransac":
            return DataProcessor._fit_ellipse_ransac(points)
        else:
            raise ValueError(f"Unknown method: {method}")

    @staticmethod
    def _filter_contour_points(points: np.ndarray, method: str, 
                             params: Dict[str, Any]) -> np.ndarray:
        #if method == "hough":
        #    return DataProcessor._filter_with_hough(points, params)
        #el
        if method == "ransac":
            return DataProcessor._filter_with_ransac_lines(points, params)
        else:
            raise ValueError(f"Unknown line filtration method: {method}")

    #Need to test
    @staticmethod
    def _filter_with_hough(points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:

        rho_res = params.get('rho_resolution', 1)
        theta_res = params.get('theta_resolution', np.pi/180)
        threshold = params.get('threshold', 50)
        min_line_length = params.get('min_line_length', 50)
        max_line_gap = params.get('max_line_gap', 10)
        angle_tolerance = params.get('angle_tolerance', 15)
        
        x_min, y_min = points.min(axis=0).astype(int)
        x_max, y_max = points.max(axis=0).astype(int)
        
        width = max(x_max - x_min + 10, 1)
        height = max(y_max - y_min + 10, 1)
        
        mask = np.zeros((height, width), dtype=np.uint8)
        contour_relative = (points - [x_min, y_min]).astype(np.int32)
        cv2.fillPoly(mask, [contour_relative], 255)
        
        lines = cv2.HoughLinesP(mask, rho_res, theta_res, threshold, 
                              minLineLength=min_line_length, maxLineGap=max_line_gap)
        
        if lines is None:
            return points
        
        edge_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            
            if (abs(angle) < angle_tolerance or 
                abs(angle - 90) < angle_tolerance or 
                abs(angle - 180) < angle_tolerance):
                edge_lines.append((x1 + x_min, y1 + y_min, x2 + x_min, y2 + y_min))
        
        if not edge_lines:
            return points
        
        distance_threshold = params.get('distance_threshold', 5.0)
        filtered_points = []
        
        for point in points:
            point_on_edge = False
            for line in edge_lines:
                x1, y1, x2, y2 = line

                distance = DataProcessor._distance_to_line(point, (x1, y1), (x2, y2))
                if distance < distance_threshold:
                    point_on_edge = True
                    break
            
            if not point_on_edge:
                filtered_points.append(point)
        
        return np.array(filtered_points)


    @staticmethod
    def _filter_with_ransac_lines(points: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        num_iterations = params.get('num_iterations', 100)
        distance_threshold = params.get('distance_threshold', 3.0)
        min_samples = params.get('min_samples', 10)
        angle_tolerance = params.get('angle_tolerance', 15)
        
        edge_lines = []
        remaining_points = points.copy()
        

        for _ in range(5):
            if len(remaining_points) < min_samples * 2:
                break
                
            best_line = None
            best_inliers = None
            best_score = 0
            
            for _ in range(num_iterations):
                indices = np.random.choice(len(remaining_points), 2, replace=False)
                p1 = remaining_points[indices[0]]
                p2 = remaining_points[indices[1]]
                

                line_angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0])) % 180
                
                if not (abs(line_angle) < angle_tolerance or 
                       abs(line_angle - 90) < angle_tolerance or 
                       abs(line_angle - 180) < angle_tolerance):
                    continue
                
                inliers = []
                for i, point in enumerate(remaining_points):
                    distance = DataProcessor._distance_to_line(point, p1, p2)
                    if distance < distance_threshold:
                        inliers.append(i)
                
                if len(inliers) > best_score and len(inliers) >= min_samples:
                    best_score = len(inliers)
                    best_line = (p1, p2)
                    best_inliers = inliers
            
            if best_line is not None:
                edge_lines.append(best_line)
                remaining_points = np.delete(remaining_points, best_inliers, axis=0)
            else:
                break
        
        if edge_lines:
            filtered_points = []
            for point in points:
                point_on_edge = False
                for line in edge_lines:
                    p1, p2 = line
                    distance = DataProcessor._distance_to_line(point, p1, p2)
                    if distance < distance_threshold:
                        point_on_edge = True
                        break
                
                if not point_on_edge:
                    filtered_points.append(point)
            
            return np.array(filtered_points)
        
        return points


    @staticmethod
    def _distance_to_line(point: np.ndarray, line_point1: np.ndarray, 
                         line_point2: np.ndarray) -> float:
        x, y = point
        x1, y1 = line_point1
        x2, y2 = line_point2
        
        numerator = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        
        if denominator == 0:
            return np.sqrt((x - x1)**2 + (y - y1)**2)
        
        return numerator / denominator

    
    @staticmethod
    def _fit_ellipse_ransac(points: np.ndarray, num_iterations: int = 100, 
                          sample_size: int = 10, threshold: float = 3.0) -> Optional[Ellipse]:

        if len(points) < 5:
            return None
        
        best_ellipse = None
        best_inliers = 0
        
        for _ in range(num_iterations):
            indices = np.random.choice(len(points), min(sample_size, len(points)), replace=False)
            sample_points = points[indices].astype(np.float32)
            
            try:
                (x, y), (major_axis, minor_axis), angle = cv2.fitEllipse(sample_points)
                candidate_ellipse = Ellipse(center=(x, y), axes=(major_axis, minor_axis), angle=angle)
                
                inlier_count = 0
                for point in points:
                    if candidate_ellipse.is_point_inside(tuple(point)):
                        inlier_count += 1
                
                if inlier_count > best_inliers:
                    best_inliers = inlier_count
                    best_ellipse = candidate_ellipse
                    
            except Exception:
                continue
        
        return best_ellipse
    
    @staticmethod
    def convert_to_processed_data(detected_data, 
                                  ellipse_method: str = "ransac",
                                  visual_debug: bool = False):
        from encapsu_view.analysis.detected_data import DetectedData
        from encapsu_view.analysis.processed_data import ProcessedData
        processed_capsules = []


        ransac_params = {
            'num_iterations': 100,
            'distance_threshold': 3.0,
            'min_samples': 30,
            'angle_tolerance': 0.5
        }
        
        for capsule in detected_data.capsules:
            contour = capsule.outer_contour

            if visual_debug:
            
                DataProcessor.visualize_contour_filtering(
                    contour=contour,
                    line_filtration_method='ransac',
                    line_filtration_params=ransac_params,
                    save_path='contour_filtering_comparison.png'
                )

                DataProcessor.compare_ellipse_fitting(
                    contour=contour,
                    line_filtration_method='ransac',
                    line_filtration_params=ransac_params,
                    save_path='ellipse_comparison.png'
                )


                DataProcessor.visualize_contour_filtering(
                    contour=contour,
                    line_filtration_method=None, 
                    save_path='original_contour.png'
                )
            


            ellipse = DataProcessor.fit_ellipse_to_contour(capsule.outer_contour, ellipse_method, line_filtration_method='ransac', line_filtration_params=ransac_params)
            
            if ellipse is not None:
                processed_capsule = ProcessedCapsule(
                    ellipse=ellipse,
                    inner_objects=capsule.inner_objects,
                    image_shape=detected_data.image_shape,
                    capsule_id=capsule.id
                )
                processed_capsules.append(processed_capsule)
        
        return ProcessedData(
            capsules=processed_capsules,
            scale=detected_data.scale,
            image_shape=detected_data.image_shape
        )