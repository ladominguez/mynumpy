from setuptools import setup, find_packages

setup(
    name='mynumpy',
    version='0.2',
    packages=find_packages(),
    install_requires=[
        'numpy',
    ],
    author='Luis A. Dominguez',
    author_email='your.email@example.com',
    description='A custom library built on top of numpy',
    url='https://github.com/ladominguez/mynumpy',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
