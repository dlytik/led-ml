from setuptools import setup, find_packages

setup(
    name='led-ml',
    version='0.1.0',
    description='Layer Extraction & Diagnostics with Machine Learning',
    long_description=open('../README.md', encoding='utf-8').read() if __name__ == '__main__' else '',
    long_description_content_type='text/markdown',
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
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.0.0',
        'matplotlib>=3.5.0',
        'pillow>=9.0.0',
    ],
)
