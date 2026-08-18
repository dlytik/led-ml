from setuptools import setup, find_packages
import os

# Get the absolute path of the directory where setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))
readme_path = os.path.join(setup_dir, "README.md")

long_desc = ""
try:
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            long_desc = f.read()
except Exception:
    pass

# Fallback text string if the README file cannot be read successfully
if not long_desc.strip():
    long_desc = "Layer-wise Energy Dissipation visualisation for LLM/ML"

setup(
    name='led-ml',
    version='0.1.5',  # Incremented to reflect the clean configuration changes
    description='Layer-wise Energy Dissipation visualisation for LLM/ML',
    long_description=long_desc,
    long_description_content_type="text/markdown",  # Valid rendering format for PyPI
    author='dlytik',
    author_email='dlytiks@gmail.com',
    url='https://github.com/dlytik/led-ml',
    project_urls={
        'Repository': 'https://github.com/dlytik/led-ml',
        'Bug Tracker': 'https://github.com/dlytik/led-ml/issues',
        'Zenodo Paper': 'https://zenodo.org/records/21967890',
    },
    packages=find_packages(),
    package_data={
        'led_ml': ['supported_models.json', 'README.md'],  # Keeps a copy bundled inside the wheel
    },
    include_package_data=True,
    python_requires='>=3.7',
    license='Apache-2.0',
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
        'Programming Language :: Python :: 3.12',
    ],
    install_requires=[
        'torch>=1.9.0',
        'transformers>=4.0.0',
        'matplotlib>=3.5.0',
        'pillow>=9.0.0',
        'urllib3<2.0.0',
    ],
)
