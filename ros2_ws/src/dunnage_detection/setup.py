from setuptools import find_packages, setup

package_name = 'dunnage_detection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['dunnage_detection/darknet_models/dunnage.cfg', 'dunnage_detection/darknet_models/dunnage.weights', 'dunnage_detection/darknet_models/dunnage.names']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rs_camera_viewer = dunnage_detection.rs_camera_viewer:main',
            'rs_bag_snapshot = dunnage_detection.rs_bag_snapshot:main',
            'bag_reader = dunnage_detection.bag_reader:main',
            'darknet_detection = dunnage_detection.darknet_detection:main'
        ],
    },
)
