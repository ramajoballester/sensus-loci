from setuptools import setup

package_name = 'ros_sensus'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools<58.3.0'
    ],
    # Currently the ros package does not take into account the
    # setuptools version, so we need to force it to be less than 58.3.0 via 
    # command line
    zip_safe=True,
    maintainer='Álvaro Ramajo-Ballester',
    maintainer_email='aramajo@pa.uc3m.es',
    description='ROS interface to sensus-loci package for environment perception',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'marker_viz = ros_sensus.marker_visualizer:main',
            'bbox_viz = ros_sensus.bbox_visualizer:main',
            'perception_manager = ros_sensus.perception_manager:main',
            'perception_visualizer = ros_sensus.perception_visualizer:main',
            'pc_publisher = ros_sensus.pc_publisher:main'
        ],
    },
)
