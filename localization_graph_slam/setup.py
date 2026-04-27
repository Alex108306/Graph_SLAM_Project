from setuptools import find_packages, setup

package_name = 'localization_graph_slam'

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
    maintainer='giang',
    maintainer_email='giang.nht108201@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'localization_graph_slam = localization_graph_slam.perform_localization:main',
            'line_extraction = localization_graph_slam.line_extraction:main',
        ],
    },
)
