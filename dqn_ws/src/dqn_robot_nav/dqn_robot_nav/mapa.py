import cv2
import numpy as np
import random
import os
import math

# --- CONFIGURACIÓN DINÁMICA ---
WORLD_SIZE = 16.0    # El mapa total es de 16m x 16m
OFFSET_X = 8.0       # Centro en 8m (mitad de 16)
OFFSET_Y = 8.0       

# Alineación pedida
ROTATION_DEG = 45.0  
SHIFT_X = -7.0
SHIFT_Y = -7.0

def test_goal_generation():
    path = os.path.expanduser('~/Desktop/dqn_robot_nav/src/dqn_robot_nav/dqn_robot_nav/cave2.png')
    img_original = cv2.imread(path)
    if img_original is None: return

    img_gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    height_px, width_px = img_gray.shape
    
    # AJUSTE CRÍTICO: Calculamos la resolución real de TU imagen
    # Esto asegura que 16 metros cubran exactamente el ancho/alto total de la imagen
    res_x = WORLD_SIZE / width_px
    res_y = WORLD_SIZE / height_px
    
    angle_rad = math.radians(ROTATION_DEG)
    
    # Tamaño del bloque del robot (0.4m convertido usando la nueva resolución)
    block_w = int(0.4 / res_x)
    block_h = int(0.4 / res_y)

    while True:
        img_display = img_original.copy()
        
        # 1. Coordenadas Odom
        gx_odom = 0
        gy_odom = 0 
        
        # 2. ROTACIÓN Y DESPLAZAMIENTO
        gx_map = (gx_odom * math.cos(angle_rad) - gy_odom * math.sin(angle_rad)) + SHIFT_X
        gy_map = (gx_odom * math.sin(angle_rad) + gy_odom * math.cos(angle_rad)) + SHIFT_Y

        # 3. CONVERSIÓN A PÍXELES USANDO ESCALA REAL DE LA IMAGEN
        px = int((gx_map + OFFSET_X) / res_x)
        py = int((gy_map + OFFSET_Y) / res_y)
        
        # 4. Invertir eje Y
        py_cv = height_px - py

        # 5. DIBUJO
        if 0 <= px < width_px and 0 <= py_cv < height_px:
            # Dibujar el bloque escalado
            cv2.rectangle(img_display, 
                          (px - block_w//2, py_cv - block_h//2), 
                          (px + block_w//2, py_cv + block_h//2), 
                          (0, 255, 0), -1)

            cv2.putText(img_display, f"Odom: {gx_odom},{gy_odom}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            print(f"Resolución calculada: {res_x:.4f} m/px")
            print(f"Odom [{gx_odom}, {gy_odom}] -> Píxel [{px}, {py_cv}]")
            cv2.imshow("Validacion Escala Real", img_display)
        else:
            print(f"Punto fuera de imagen: Map[{gx_map:.2f}, {gy_map:.2f}] -> Px[{px}, {py_cv}]")

        key = cv2.waitKey(0) & 0xFF
        if key == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_goal_generation()
