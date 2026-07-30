from setuptools import find_packages, setup

package_name = 'haptic_plume_experiment'

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
    description=('Experiment harness: trial lifecycle, condition gating, '
                 'and offline estimator evaluation'),
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'eval_estimator = haptic_plume_experiment.eval_estimator:main',
        ],
    },
)
