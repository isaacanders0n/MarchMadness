from setuptools import find_packages, setup

setup(
    name="projectDag",
    packages=find_packages(exclude=["projectDag_tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud"
    ],
    extras_require={"dev": ["dagster-webserver", "pytest"]},
)
