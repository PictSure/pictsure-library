from setuptools import setup, find_packages

setup(
    name='PictSure',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=2.7.0',
        'torchvision>=0.22.0',
        'numpy>=1.26.4',
        'Pillow',  # Version not specified as it's a dependency of torchvision
        'click>=8.1.7',
        'tqdm>=4.66.4',
        'requests>=2.32.3'
    ],
    entry_points={
        'console_scripts': [
            'pictsure=PictSure.cli:cli',
        ],
    },
    author='Cornelius Wolff, Lukas Schiesser',
    author_email='lukas.schiesser@dfki.de',
    description='A package for generalized image classification with PyTorch.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://git.ni.dfki.de/pictsure/pictsure-library',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
