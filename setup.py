from setuptools import setup, find_packages

setup(
    name="io_eai_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pybullet",
        "viser",
        "opencv-python",
        "plyfile",
    ],
    author="Feiyu Zhao",
    author_email="zhaofy@io-ai.tech",
    description="The goal of this project is to provide a low-cost, highly controllable, high-fidelity simulation environment for embodied AI.",
)
