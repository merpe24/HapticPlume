from setuptools import find_packages, setup

package_name = 'haptic_plume_gas_sim'

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
    description=('Truth side of the gas sim: analytic gas field plus the lagged, '
                 'noisy chemical-sensor model'),
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
