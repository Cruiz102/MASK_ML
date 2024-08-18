from setuptools import setup, find_packages
REQUIRED_PACKAGES = [
    'hydra-core==1.3.2'
]
setup(
    name="ml_mask",
    version="0.1",
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    python_requires=">=3.10.12",
)
