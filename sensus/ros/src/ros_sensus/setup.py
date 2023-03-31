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
    zip_safe=True,
    maintainer='Álvaro Ramajo-Ballester',
    maintainer_email='aramajo@pa.uc3m.es',
    description='ROS interface to sensus-loci package for environment perception',
    license='GPL-3.0-only',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
