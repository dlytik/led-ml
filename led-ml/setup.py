from setuptools import setup, find_packages

setup(
    name='led-ml',
    # 0.1.0 is the correct and standard starting version for an initial development release
    version='0.1.0',
    description='Layer-wise Energy Dissipation visualisation for LLM/ML',
    # Reads the README documentation from the parent directory for PyPI rendering
    long_description=open('../README.md', encoding='utf-8').read() if __name__ == '__main__' else '',
    long_description_content_type='text/markdown',
    author='dlytik',
    author_email='dlytiks@gmail.com',
    url='https://github.com/dlytik/led-ml',
    project_urls={
        'Repository': 'https://github.com/dlytik/led-ml',
        'Bug Tracker': 'https://github.com/dlytik/led-ml/issues',
    },
    packages=find_packages(),
    package_data={
        'led_ml': ['supported_models.json'],
    },
    include_package_data=True,
    python_requires='>=3.7',
    license='Apache-2.0',
    # Classifiers help PyPI properly catalog and display your package attributes
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.0.0',
        'matplotlib>=3.5.0',
        'pillow>=9.0.0',
    ],
)
