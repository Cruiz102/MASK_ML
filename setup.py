from setuptools import setup, find_packages
REQUIRED_PACKAGES = [
    'hydra-core==1.3.2',
    'pycocotools==2.0.7',
    'opencv-python==4.10.0.84',
    'torch==2.4.0',
    'torchvision==0.19.0',
    'hydra-optuna-sweeper'
]
setup(
    name="ml_mask",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.12",
)
# pip uninstall opencv-python-headless opencv-python
# pip install opencv-python
