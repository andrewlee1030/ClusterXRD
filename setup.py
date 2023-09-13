from setuptools import setup, find_packages

setup(name='ClusterXRD',
    version='1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires = ['matplotlib', 'scikit-learn', 'numpy','pandas'],
    author='Andrew S. Lee',
    author_email='andrewlee1030@gmail.com',
    python_requires='>=3.11',
    )