from setuptools import setup, find_packages

setup(
    name='nodos',
    version='0.1.0',
    author='Nodos',
    description='A Python interface for the Nodos.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mediaz/nos-modules',  # Replace with your repo URL
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
    ]
)
