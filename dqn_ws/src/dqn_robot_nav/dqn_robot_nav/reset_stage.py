import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from geometry_msgs.msg import Quaternion
import math

class OdomResetWrapper(Node):
    def __init__(self):
        super().__init__('odom_reset_wrapper')

        # Configuración de tópicos
        self.input_odom_topic = '/odom'
        self.output_odom_topic = '/odom/sim'
        self.stage_reset_service = '/reset_positions' 
        
        # Offsets iniciales
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.offset_yaw = 0.0
        
        # Estados de control
        self.reset_requested = False
        self.frames_to_skip = 0

        self.sub_odom = self.create_subscription(Odometry, self.input_odom_topic, self.odom_callback, 10)
        self.pub_odom = self.create_publisher(Odometry, self.output_odom_topic, 10)

        # Tu servicio de reset
        self.srv_reset = self.create_service(Empty, 'reset_sim', self.reset_sim_callback)
        self.client_stage = self.create_client(Empty, self.stage_reset_service)

        self.get_logger().info("Odom Reset Wrapper: Monitoreando offsets...")

    def reset_sim_callback(self, request, response):
        """Paso 1: Se solicita el reset"""
        self.get_logger().info("Recibida petición de Reset Sim")
        
        if self.client_stage.wait_for_service(timeout_sec=1.0):
            # Llamamos a Stage para mover al robot físicamente
            self.client_stage.call_async(Empty.Request())
            
            # Activamos la bandera para capturar el NUEVO offset en el callback
            self.reset_requested = True
            # Saltamos 10 frames para asegurar que Stage ya movió al robot
            self.frames_to_skip = 10 
        
        return response

    def odom_callback(self, msg: Odometry):
        # 1. Extraer datos crudos (los que NO vuelven a cero)
        raw_x = msg.pose.pose.position.x
        raw_y = msg.pose.pose.position.y
        _, _, raw_yaw = self.euler_from_quaternion(msg.pose.pose.orientation)

        # 2. Lógica de captura de Offset
        if self.reset_requested:
            if self.frames_to_skip > 0:
                self.frames_to_skip -= 1
                return # Esperamos a que el robot se teletransporte
            
            # Capturamos la posición actual de Stage como el nuevo "Cero"
            self.offset_x = raw_x
            self.offset_y = raw_y
            self.offset_yaw = raw_yaw
            self.reset_requested = False
            self.get_logger().info(f"OdomSim REINICIADA. Offset capturado en: x={raw_x:.2f}, y={raw_y:.2f}")

        # 3. Transformación a Coordenadas Relativas (Matriz de Rotación Inversa)
        # Calculamos la diferencia respecto al punto de reset
        dx = raw_x - self.offset_x
        dy = raw_y - self.offset_y
        
        # Rotamos el vector para que coincida con el nuevo frente (Yaw = 0)
        cos_inv = math.cos(-self.offset_yaw)
        sin_inv = math.sin(-self.offset_yaw)
        
        sim_x = dx * cos_inv - dy * sin_inv
        sim_y = dx * sin_inv + dy * cos_inv
        sim_yaw = raw_yaw - self.offset_yaw

        # Normalizar Yaw a [-pi, pi]
        sim_yaw = math.atan2(math.sin(sim_yaw), math.cos(sim_yaw))

        # 4. Publicar en /odom/sim
        new_msg = Odometry()
        new_msg.header = msg.header
        new_msg.header.frame_id = "odom"
        new_msg.child_frame_id = "base_link"
        
        new_msg.pose.pose.position.x = sim_x
        new_msg.pose.pose.position.y = sim_y
        new_msg.pose.pose.orientation = self.quaternion_from_euler(0, 0, sim_yaw)
        new_msg.twist = msg.twist # La velocidad no cambia

        self.pub_odom.publish(new_msg)

    def euler_from_quaternion(self, q):
        t3 = +2.0 * (q.w * q.z + q.x * q.y)
        t4 = +1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return 0, 0, math.atan2(t3, t4)

    def quaternion_from_euler(self, roll, pitch, yaw):
        qz = math.sin(yaw/2)
        qw = math.cos(yaw/2)
        return Quaternion(x=0.0, y=0.0, z=qz, w=qw)

def main(args=None):
    rclpy.init(args=args)
    node = OdomResetWrapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()