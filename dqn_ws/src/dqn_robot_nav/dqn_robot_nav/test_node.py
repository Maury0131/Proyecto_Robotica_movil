#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env
import numpy as np
import time
from threading import Thread
import cv2
import torch
import sys

class DQNTestNode(Node):
    """
    Nodo de evaluación para el agente DQN.
    Carga un modelo .pth y lo ejecuta sin entrenamiento (Greedy Policy).
    """

    def __init__(self, model_path: str):
        super().__init__('dqn_test_node')

        # ======================================================
        # CONFIGURACIÓN (Debe coincidir con training)
        # ======================================================
        self.n_bins = 24
        self.state_size = self.n_bins + 2  # LiDAR (24) + Dist + Angle
        self.action_size = 5

        # Visualización
        self.viz_window_name = "TEST - Navegacion DQN"

        # ==============================
        # 1. Inicializar Entorno
        # ==============================
        self.env = TurtleBot3Env()
        
        # IMPORTANTE: Para el test, ponemos la dificultad máxima (Mapa Completo)
        self.env.max_goal_dist = 16.0 
        self.get_logger().info("Configuración de Test: Rango de meta COMPLETO (16.0m)")

        # ==============================
        # 2. Inicializar Agente y Cargar Modelo
        # ==============================
        self.agent = DQNAgent(
            state_size=self.state_size, 
            action_size=self.action_size,
            epsilon_start=0.0, # Greedy puro: solo hace lo que sabe
            epsilon_min=0.0
        )
        
        try:
            self.agent.load(model_path)
            self.agent.epsilon = 0.0  # Aseguramos que sea 0
            self.get_logger().info(f"✅ Modelo cargado exitosamente desde: {model_path}")
        except Exception as e:
            self.get_logger().error(f"❌ Error al cargar el modelo: {e}")
            sys.exit(1)

        # ==============================
        # 3. Hilo de ROS (Sensores)
        # ==============================
        self.ros_thread = Thread(target=lambda: rclpy.spin(self.env))
        self.ros_thread.daemon = True
        self.ros_thread.start()

    def test(self, n_episodes: int = 20):
        successes = 0
        total_rewards = []
        steps_list = []

        self.get_logger().info(f"\n🚀 Iniciando evaluación de {n_episodes} episodios...\n")

        for episode in range(1, n_episodes + 1):
            
            # Reset del entorno
            state = self.env.reset(random_goal=True)
            
            # Esperar a que lleguen datos del LiDAR si es necesario
            while self.env.scan_data is None:
                time.sleep(0.05)

            episode_reward = 0.0
            done = False
            success = False
            
            # Bucle de pasos (Step loop)
            for step in range(1000): # Max 1000 pasos para evitar loops infinitos
                
                # 1. Acción (Training=False usa epsilon=0, o sea, pura predicción)
                action = self.agent.act(state, training=False)

                # 2. Paso en el entorno (¡OJO! Ahora devuelve 4 valores)
                next_state, reward, done, info = self.env.step(action)

                # 3. Visualización
                img = self.env.render_map()
                if img is not None:
                    # Añadir texto en la imagen
                    cv2.putText(img, f"Ep: {episode}/{n_episodes}", (10, 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                    cv2.imshow(self.viz_window_name, img)
                    cv2.waitKey(1)

                episode_reward += reward
                state = next_state

                if done:
                    # Usamos la bandera 'success' que programamos en el Environment
                    if info.get("success", False):
                        successes += 1
                        success = True
                        self.get_logger().info(f"✅ Ep {episode}: ÉXITO | Pasos: {step} | Reward: {episode_reward:.1f}")
                    else:
                        self.get_logger().info(f"❌ Ep {episode}: FALLO (Colisión/Timeout) | Pasos: {step} | Reward: {episode_reward:.1f}")
                    break
            
            total_rewards.append(episode_reward)
            steps_list.append(step)

        # ==============================
        # Reporte Final
        # ==============================
        cv2.destroyAllWindows()
        success_rate = (successes / n_episodes) * 100.0
        avg_reward = np.mean(total_rewards)
        avg_steps = np.mean(steps_list)

        self.get_logger().info("\n" + "=" * 50)
        self.get_logger().info(f"📊 RESULTADOS FINALES ({n_episodes} episodios)")
        self.get_logger().info(f"   Tasa de Éxito:      {success_rate:.1f}%")
        self.get_logger().info(f"   Recompensa Promedio: {avg_reward:.2f}")
        self.get_logger().info(f"   Pasos Promedio:      {avg_steps:.1f}")
        self.get_logger().info("=" * 50)


def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("⚠️  USO INCORRECTO")
        print("   Ejecuta: ros2 run dqn_robot_nav test_node <ruta_al_modelo.pth>")
        return

    model_path = sys.argv[1]
    tester = DQNTestNode(model_path)

    try:
        # Ejecutar 20 episodios de prueba
        tester.test(n_episodes=20)
    except KeyboardInterrupt:
        print("\n[!] Test interrumpido por el usuario")
    finally:
        tester.env.send_velocity(0.0, 0.0)
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()