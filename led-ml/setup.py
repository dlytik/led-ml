from setuptools import setup, find_packages
import os
# Get the absolute path of the directory where setup.py is located
setup_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to target the repository root directory
parent_dir = os.path.dirname(setup_dir)

# Read the single source-of-truth README.md from the root folder safely
long_desc = ""
readme_path = os.path.join(parent_dir, "README.md")

try:
    if os.path.exists(readme_path):
        with open(readme_path, encoding="utf-8") as f:
            long_desc = f.read()
    else:
        # Fallback to local folder if the parent directory structure differs
        local_readme = os.path.join(setup_dir, "README.md")
        if os.path.exists(local_readme):
            with open(local_readme, encoding="utf-8") as f:
                long_desc = f.read()
except Exception:
    long_desc = "Layer-wise Energy Dissipation visualisation for LLM/ML"
setup(
    name='led-ml',
    # 0.1.0 is the correct and standard starting version for an initial development release
    version='0.1.3',
    description='Layer-wise Energy Dissipation visualisation for LLM/ML',
    # Reads the README documentation from the parent directory for PyPI rendering
    long_description=long_desc,
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
        'led_ml': ['supported_models.json'],
    },
    long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
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
        'urllib3<2.0.0',
    ],
)
