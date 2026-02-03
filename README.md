# Proyecto_Robotica_movil
El ws usado para el proyecto final con la simulacion en stage de un robot diferencial y con un dqn agent  
A continuacion estan los pasos para ejecutar el siguiente ws:  
primero descargue y descomprimir el archivo.  
Previamenbte debe haber descargado Stage, una vez instalado corra la siguiente linea:  
ros2 launch stage_ros2 demo.launch.py world:=cave use_stamped_velocity:=false  
asegurese que este el robot con los 360 grados del lidar, en caso de no estarlo puede modificarlo en el archivo:  
stage_ros2/world/include/robots.inc y modificar los grados del lidar  
Ahora dirigirse al archivo descargado hasta Proyecto_Robotica_movil-main/dqn_ws y hacer:  
colcon build   
source install/setup.bash
Luego debemos correr el nodo de restart para el entrenamiento con :  
ros2 run dqn_robot_nav reset_stage  
En una nueva terminal siguiendo los pasos de compliar el ws para el entrenamiento ejecutaremos lo siguiente  
ros2 run dqn_robot_nav train_node   
Esto comenzara el entrenamiento, podra ver en la simulacion de stage y una imagen donde muestra la posicion odom y el goal generado para la epoca de entrenamiento, el codigo por defecto  
entrenara 1000 epocas e ira guardando el modelo cada 100 epocas.  
Una vez terminado el entrenamiento para probar algun modelo se usara lo siguiente:  
ros2 run dqn_robot_nav test_node /home/mau/Desktop/dqn_robot_nav/results_20260131_224605/model_ep500.pth  
Tomar nota que debera cambiar la direccion con la suya donde se guardo el modelo.  
