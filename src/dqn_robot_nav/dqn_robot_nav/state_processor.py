import numpy as np
import math
from typing import Tuple

class StateProcessor:
    """Procesa datos LiDAR 360° y odometría para la red DQN en PyTorch"""
    
    def __init__(self, n_lidar_bins: int = 24):
        self.n_lidar_bins = n_lidar_bins
        # Rango físico del Lidar en Stage (ajustado a tu configuración)
        self.max_lidar_range = 3.5  
        
    def process_lidar(self, scan_data: np.ndarray) -> np.ndarray:
        # Manejo de datos nulos
        if scan_data is None or len(scan_data) == 0:
            return np.ones(self.n_lidar_bins, dtype=np.float32)

        # Copia para no modificar el original por referencia
        scan_array = np.array(scan_data, copy=True)
        
        # Limpieza de datos
        scan_array[np.isinf(scan_array)] = self.max_lidar_range
        scan_array[np.isnan(scan_array)] = self.max_lidar_range
        scan_array = np.clip(scan_array, 0, self.max_lidar_range)
        
        num_points = len(scan_array)
        
        # Manejo de sensores 360°: dividimos el array en sectores iguales
        # y tomamos el mínimo de cada uno para detectar el obstáculo más cercano
        binned_scan = []
        sector_size = num_points / self.n_lidar_bins
        
        for i in range(self.n_lidar_bins):
            start_idx = int(i * sector_size)
            end_idx = int((i + 1) * sector_size)
            
            # Evitamos sectores vacíos
            sector = scan_array[start_idx:end_idx]
            if len(sector) > 0:
                binned_scan.append(np.min(sector))
            else:
                binned_scan.append(self.max_lidar_range)
        
        # Normalización [0, 1] - Importante para la estabilidad de la red neuronal
        return np.array(binned_scan, dtype=np.float32) / self.max_lidar_range
    
    def compute_goal_info(self, 
                         current_pos: Tuple[float, float],
                         goal_pos: Tuple[float, float],
                         current_yaw: float) -> np.ndarray:
        
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        
        distance = math.sqrt(dx**2 + dy**2)
        
        # Ángulo relativo: meta - orientación actual
        angle_to_goal = math.atan2(dy, dx)
        relative_angle = angle_to_goal - current_yaw
        
        # Normalizar ángulo a [-π, π] usando atan2(sin, cos)
        relative_angle = math.atan2(math.sin(relative_angle), math.cos(relative_angle))
        
        # --- NORMALIZACIÓN ---
        # Distancia: 8.0m es un buen límite para mapas típicos de Stage
        distance_norm = np.clip(distance / 8.0, 0, 1)  
        
        # Ángulo: De [-π, π] a [-1, 1]
        angle_norm = relative_angle / math.pi 
        
        return np.array([distance_norm, angle_norm], dtype=np.float32)
    
    def get_state(self,
                  scan_data: np.ndarray,
                  current_pos: Tuple[float, float],
                  goal_pos: Tuple[float, float],
                  current_yaw: float) -> np.ndarray:
        """
        Retorna el vector de estado concatenado (Ej: 24 + 2 = 26 elementos)
        """
        lidar_state = self.process_lidar(scan_data)
        goal_state = self.compute_goal_info(current_pos, goal_pos, current_yaw)
        
        # Salida final lista para torch.as_tensor()
        return np.concatenate([lidar_state, goal_state]).astype(np.float32)