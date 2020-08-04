from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='onnxt5',
    version='0.1.3',
    license='apache-2.0',
    description='Blazing fast summarization, translation, text-generation, Q&A and more using T5 in ONNX',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/abelriboulot/onnxt5',
    download_url = 'https://github.com/abelriboulot/onnxt5/archive/0.0.3.tar.gz',
    author='Abel Riboulot',
    author_email='abel@kta.io',
    keywords = ['T5', 'ONNX', 'onnxruntime', 'NLP', 'transformer', 'generate text', 'summarization', 'translation',
                'q&a', 'machine learning', 'inference', 'fast inference'],
    packages=find_packages(),  # Required
    python_requires='>=3.5, <4',
    install_requires=[
        "onnxruntime>=1.4.0",
        "requests>=2.22.0",
        "torch>=1.4.0",
        "tqdm>=4.48.2",
        "transformers>=3.0.2",
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
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
      ],
)
