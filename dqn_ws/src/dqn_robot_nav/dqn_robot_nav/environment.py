import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import numpy as np
import math
import time
import cv2
import os

class TurtleBot3Env(Node):
    def __init__(self):
        super().__init__('turtlebot3_env')
        
        # --- Configuración de Tópicos y Servicios ---
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/base_scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom/sim', self.odom_callback, 10)
        self.reset_sim_client = self.create_client(Empty, 'reset_sim')
        
        # --- CONFIGURACIÓN DINÁMICA DEL MUNDO ---
        self.WORLD_SIZE = 16.0
        self.OFFSET_X = 8.0
        self.OFFSET_Y = 8.0
        self.ROTATION_DEG = 45.0
        self.SHIFT_X = -7.0
        self.SHIFT_Y = -7.0
        
        # --- CURRICULUM LEARNING ---
        # Esta variable se modifica desde el nodo de entrenamiento (dqn_training_node)
        self.max_goal_dist = 3.5 
        
        # --- MEMORIA (Anti-Camping) ---
        self.position_history = [] 
        self.max_history_len = 100 # ~5 segundos (si el step dura 0.05s)
        
        # --- CARGA DEL MAPA ---
        # Ajusta la ruta si es necesario
        path = os.path.expanduser('~/Escritorio/dqn_robot_nav/src/dqn_robot_nav/dqn_robot_nav/cave2.png')
        self.img_original = cv2.imread(path)
        
        if self.img_original is None:
            self.get_logger().error(f"CRÍTICO: No se encontró la imagen en {path}")
            self.img_gray = None
            self.height_px, self.width_px = 0, 0
        else:
            self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            self.height_px, self.width_px = self.img_gray.shape
            self.res_x = self.WORLD_SIZE / self.width_px
            self.res_y = self.WORLD_SIZE / self.height_px
            self.block_w = int(0.4 / self.res_x)
            self.block_h = int(0.4 / self.res_y)

        # Estado inicial
        self.scan_data = None
        self.position = [0.0, 0.0]
        self.yaw = 0.0
        self.goal_position = [0.0, 0.0]
        self.last_distance = 0.0
        
        # Acciones: 0:Adelante, 1:Izq, 2:Der, 3:Adelante+Izq, 4:Adelante+Der
        self.actions = {0: (0.22, 0.0), 1: (0.0, 0.8), 2: (0.0, -0.8), 3: (0.15, 0.4), 4: (0.15, -0.4)}

    # ----------------------------------------------------------------
    # TRANSFORMACIONES DE COORDENADAS
    # ----------------------------------------------------------------
    def get_pixel_coords(self, gx_odom, gy_odom):
        angle_rad = math.radians(self.ROTATION_DEG)
        gx_map = (gx_odom * math.cos(angle_rad) - gy_odom * math.sin(angle_rad)) + self.SHIFT_X
        gy_map = (gx_odom * math.sin(angle_rad) + gy_odom * math.cos(angle_rad)) + self.SHIFT_Y
        px = int((gx_map + self.OFFSET_X) / self.res_x)
        py = int((gy_map + self.OFFSET_Y) / self.res_y)
        py_cv = self.height_px - py
        return px, py_cv

    def get_odom_coords(self, px, py_cv):
        py = self.height_px - py_cv
        gx_map = (px * self.res_x) - self.OFFSET_X
        gy_map = (py * self.res_y) - self.OFFSET_Y
        delta_x = gx_map - self.SHIFT_X
        delta_y = gy_map - self.SHIFT_Y
        angle_rad = math.radians(self.ROTATION_DEG)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        gx_odom = delta_x * cos_a + delta_y * sin_a
        gy_odom = -delta_x * sin_a + delta_y * cos_a
        return gx_odom, gy_odom

    # ----------------------------------------------------------------
    # GENERACIÓN DE METAS
    # ----------------------------------------------------------------
    def generate_valid_goal(self):
        if self.img_gray is None: return

        valid = False
        intentos = 0
        
        while not valid:
            intentos += 1
            px = np.random.randint(self.block_w, self.width_px - self.block_w)
            py_cv = np.random.randint(self.block_h, self.height_px - self.block_h)

            x1 = px - self.block_w // 2
            y1 = py_cv - self.block_h // 2
            x2 = px + self.block_w // 2
            y2 = py_cv + self.block_h // 2
            roi = self.img_gray[y1:y2, x1:x2]
            
            if np.all(roi > 250):
                gx_odom, gy_odom = self.get_odom_coords(px, py_cv)
                dist = math.sqrt((gx_odom - self.position[0])**2 + (gy_odom - self.position[1])**2)
                
                # Usamos self.max_goal_dist (Curriculum Learning)
                if 1.5 < dist <= self.max_goal_dist:
                    self.goal_position = [gx_odom, gy_odom]
                    valid = True

            if intentos > 10000:
                self.get_logger().warn("No se pudo encontrar meta válida en el rango actual")
                break

    # ----------------------------------------------------------------
    # RESET
    # ----------------------------------------------------------------
    def reset(self, random_goal=True):
        self.send_velocity(0.0, 0.0)
        
        if self.reset_sim_client.wait_for_service(timeout_sec=1.0):
            self.reset_sim_client.call_async(Empty.Request())
        
        time.sleep(0.6)
        
        # Limpiar la memoria al reiniciar
        self.position_history = []
        
        for _ in range(10): 
            rclpy.spin_once(self, timeout_sec=0.01)
            
        if random_goal: 
            self.generate_valid_goal()
            
        self.last_distance = self.distance_to_goal()
        return self.get_state()

    # ----------------------------------------------------------------
    # STEP (AQUÍ ESTÁ LA MAGIA)
    # ----------------------------------------------------------------
    def step(self, action):
        v, w = self.actions[action]
        self.send_velocity(v, w)
        
        time.sleep(0.05)
        rclpy.spin_once(self, timeout_sec=0)
        
        done = False
        dist = self.distance_to_goal()
        
        # 1. ACTUALIZAR HISTORIAL (Anti-Camping)
        self.position_history.append(self.position)
        if len(self.position_history) > self.max_history_len:
            self.position_history.pop(0)

        # 2. CALCULAR CASTIGO POR ATASCAMIENTO
        stuck_penalty = 0.0
        if len(self.position_history) == self.max_history_len:
            # Distancia neta recorrida en los últimos 5 segundos
            old_x, old_y = self.position_history[0]
            curr_x, curr_y = self.position
            dist_moved = math.sqrt((curr_x - old_x)**2 + (curr_y - old_y)**2)
            
            # Si se movió menos de 1.0m en 100 pasos -> Castigo fuerte
            if dist_moved < 1.0:
                stuck_penalty = -2.0 

        # 3. CALCULAR CASTIGO POR ROTACIÓN (Anti-Spinning)
        rotation_penalty = 0.0
        # Si la velocidad lineal es muy baja Y hay rotación
        if v < 0.05 and abs(w) > 0.0:
            rotation_penalty = -0.2 # Impuesto al mareo

        # 4. RECOMPENSA FINAL
        reached_goal = False
        
        if self.is_collision():
            reward = -100.0
            done = True
        elif dist < 0.45: # Puedes subir esto a 0.50 si sigue fallando por poco
            reward = 200.0
            done = True
            reached_goal = True
        else:
            # Fórmula maestra: Progreso + Tiempo + Atascamiento + Rotación
            # Time penalty fijo: -0.2
            shaping = (self.last_distance - dist) * 30.0
            reward = shaping - 0.2 + stuck_penalty + rotation_penalty
            
            self.last_distance = dist
            
        # Devolvemos 4 valores
        return self.get_state(), reward, done, {"success": reached_goal}

    # ----------------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------------
    def render_map(self):
        if self.img_original is None: return None
        img_display = self.img_original.copy()
        px, py_cv = self.get_pixel_coords(self.goal_position[0], self.goal_position[1])
        cv2.rectangle(img_display, (px-self.block_w//2, py_cv-self.block_h//2), 
                      (px+self.block_w//2, py_cv+self.block_h//2), (0, 255, 0), -1)
        rx, ry = self.get_pixel_coords(self.position[0], self.position[1])
        cv2.circle(img_display, (rx, ry), 5, (255, 0, 0), -1)
        return img_display

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 3.5
        self.scan_data = ranges

    def odom_callback(self, msg):
        self.position = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1-2*(q.y*q.y + q.z*q.z))

    def is_collision(self):
        if self.scan_data is None: return False
        return np.min(self.scan_data) < 0.20

    def distance_to_goal(self):
        return math.sqrt((self.goal_position[0]-self.position[0])**2 + (self.goal_position[1]-self.position[1])**2)

    def get_state(self):
        if self.scan_data is None: return np.zeros(26, dtype=np.float32)
        idx = np.linspace(0, len(self.scan_data)-1, 24, dtype=int)
        lidar = self.scan_data[idx] / 3.5
        dist = self.distance_to_goal() / 12.0 
        dx, dy = self.goal_position[0]-self.position[0], self.goal_position[1]-self.position[1]
        target_angle = math.atan2(dy, dx)
        rel_angle = math.atan2(math.sin(target_angle - self.yaw), math.cos(target_angle - self.yaw)) / math.pi
        return np.append(lidar, [dist, rel_angle]).astype(np.float32)

    def send_velocity(self, v, w):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_vel_pub.publish(msg)
