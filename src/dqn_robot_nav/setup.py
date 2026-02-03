from setuptools import find_packages, setup

package_name = 'dqn_robot_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='tealfox',
    maintainer_email='crimzonfox978@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'train_node = dqn_robot_nav.train_node:main',
            'dqn_agent = dqn_robot_nav.dqn_agent:main',
            'state_processor = dqn_robot_nav.state_processor:main',
            'environment = dqn_robot_nav.environment:main',
            'reset_stage = dqn_robot_nav.reset_stage:main',
            'test_node = dqn_robot_nav.test_node:main'
        ],
    },
)
