from setuptools import find_packages, setup

package_name = 'haptic_plume_estimation'

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
    maintainer='premmm',
    maintainer_email='prem.pannavit@gmail.com',
    description=('MultiPLE estimator: sensor-lag compensation, particle filter, '
                 'plume consuming, clustering, MI waypoint scoring'),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
