#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from threading import Thread
import time
import cv2

# Import Agent & Environment
from dqn_robot_nav.dqn_agent import DQNAgent
from dqn_robot_nav.environment import TurtleBot3Env


class DQNTrainingNode(Node):
    def __init__(self):
        super().__init__('dqn_training_node')

        # ==============================
        # State / Action configuration
        # ==============================
        self.n_bins = 24
        self.state_size = self.n_bins + 2   # lidar + [dist_to_goal, angle_to_goal]
        self.action_size = 5

        # ==============================
        # Training parameters
        # ==============================
        self.n_episodes = 1000
        self.max_steps_per_episode = 1000

        # Environment
        self.env = TurtleBot3Env()

        # ==============================
        # DQN Agent
        # ==============================
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=self.action_size,
            learning_rate=0.001,   # Subimos LR para compensar rewards bajos
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.99,   # decay per EPISODE
            memory_size=30000,
            batch_size=128,
            target_update_freq=500
        )

        self.episode_rewards = []
        self.episode_steps = []
        self.success_count = 0

        # Results directory
        self.results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)

        # ROS spinning thread (sensors only)
        self.ros_thread = Thread(target=lambda: rclpy.spin(self.env))
        self.ros_thread.daemon = True
        self.ros_thread.start()

    # ==============================
    # Training Loop
    # ==============================
    def train(self):
        device_info = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        )

        self.get_logger().info(f"🚀 Entrenamiento iniciado en: {device_info}")

        for episode in range(1, self.n_episodes + 1):

            # ============================================================
            # 🎓 CURRICULUM LEARNING (MODO PROGRESIVO)
            # ============================================================
            if episode < 100:
                # FASE 1: Kinder (Distancia corta)
                self.env.max_goal_dist = 3.5
                if episode == 1: 
                    self.get_logger().info("--- 🟢 FASE 1: INICIANDO EN RANGO CORTO (3.5m) ---")
                    
            elif episode < 300:
                # FASE 2: Escuela (Distancia media)
                self.env.max_goal_dist = 7.0
                
                # Boost de Epsilon al cambiar de fase para que explore el nuevo rango
                if episode == 100:
                    self.get_logger().info("--- 🟡 FASE 2: AUMENTANDO RANGO A 7.0m (+Boost Eps) ---")
                    self.agent.epsilon = max(self.agent.epsilon, 0.5)

            else:
                # FASE 3: Vida Real (Mapa completo)
                self.env.max_goal_dist = 16.0
                
                if episode == 300:
                    self.get_logger().info("--- 🔴 FASE 3: MAPA COMPLETO 16.0m (+Boost Eps) ---")
                    self.agent.epsilon = max(self.agent.epsilon, 0.3)
            # ============================================================

            # Reset environment with RANDOM GOAL (respecting max_goal_dist)
            state = self.env.reset(random_goal=True)

            # Wait for LiDAR
            while self.env.scan_data is None:
                time.sleep(0.05)

            episode_reward = 0.0

            for step in range(self.max_steps_per_episode):

                # 1. Agent action
                action = self.agent.act(state, training=True)

                # 2. Environment step
                # OJO: Ahora recibimos 4 valores incluyendo 'info'
                next_state, reward, done, info = self.env.step(action)

                # 3. Visualization (safe)
                if step % 5 == 0:
                    img = self.env.render_map()
                    if img is not None:
                        cv2.imshow("Monitoreo DQN - Stage", img)
                        cv2.waitKey(1)

                # 4. Store experience
                self.agent.remember(state, action, reward, next_state, done)

                # 5. Learn
                if len(self.agent.memory) >= self.agent.batch_size:
                    self.agent.replay()

                episode_reward += reward
                state = next_state

                if done:
                    # Usamos la bandera del environment, más seguro que mirar los puntos
                    if info.get("success", False):  
                        self.success_count += 1
                    break

            # ==============================
            # Episode end
            # ==============================
            self.agent.decay_epsilon()

            self.episode_rewards.append(episode_reward)
            self.episode_steps.append(step + 1)

            # Logging
            if episode % 5 == 0:
                success_rate = (self.success_count / episode) * 100.0
                self.get_logger().info(
                    f"Ep: {episode} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Eps: {self.agent.epsilon:.3f} | "
                    f"Éxito: {success_rate:.1f}% | "
                    f"Rango: {self.env.max_goal_dist}m"
                )

            # Save checkpoints
            if episode % 100 == 0:
                self.agent.save(
                    os.path.join(self.results_dir, f"model_ep{episode}.pth")
                )
                self.plot_results(f"progress_ep{episode}.png")

        self.plot_results("final_metrics.png")

    # ==============================
    # Plot metrics
    # ==============================
# ... (dentro de la clase DQNTrainingNode)

    def plot_results(self, filename):
        # 1. Gráfica estándar (Reward y Steps)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards, color='blue', alpha=0.3)
        plt.title("Recompensa por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Reward")

        plt.subplot(1, 2, 2)
        plt.plot(self.episode_steps, color='orange')
        plt.title("Pasos por Episodio")
        plt.xlabel("Episodio")
        plt.ylabel("Steps")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, filename))
        plt.close()

        # 2. NUEVA: Gráfica de Reward Acumulado (Media Móvil)
        if len(self.episode_rewards) > 10:
            plt.figure(figsize=(10, 6))
            
            # Calcular media móvil de los últimos 20 episodios
            window = 20
            avg_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
            
            plt.plot(range(window, len(avg_rewards) + window), avg_rewards, color='red', linewidth=2)
            plt.fill_between(range(window, len(avg_rewards) + window), avg_rewards, color='red', alpha=0.1)
            
            plt.title(f"Recompensa Media Acumulada (Ventana: {window})")
            plt.xlabel("Episodio")
            plt.ylabel("Reward Promedio")
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Guardar con nombre específico
            acc_filename = filename.replace("progress_", "accumulated_").replace("final_", "final_accumulated_")
            plt.savefig(os.path.join(self.results_dir, acc_filename))
            plt.close()


def main(args=None):
    rclpy.init(args=args)
    trainer = DQNTrainingNode()

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[!] Interrupción manual")
    finally:
        trainer.env.send_velocity(0.0, 0.0)
        final_model = os.path.join(trainer.results_dir, "model_final.pth")
        trainer.agent.save(final_model)
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
