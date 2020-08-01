from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='onnxt5',
    version='0.0.1',
    license='Apache',
    description='T5 Implementation in ONNX with utility functions for fast inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abelriboulot/onnxt5',
    author='Abel Riboulot',
    author_email='abel@kta.io',
    package_dir={'': 'onnxt5'},
    packages=find_packages(where='onnxt5'),  # Required
    python_requires='>=3.5, <4',
    install_requires=[
        "transformers>=3.0.2",
        "onnxruntime>=1.4.0",
        "boto3>=1.14.0",
        "torch>=1.4.0"
    ],
    extras_require={
        'dev': ["unittest"],
        'test': [],
    },

    package_data={
        'sample': [],
    },

    entry_points={
    },

    project_urls={
        'Repo': 'https://github.com/abelriboulot/onnxt5',
    },
    classifiers=[
        'Development Status :: 2 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
)