from setuptools import setup, find_packages
setup(
    name="autoeit",
    version="1.0.0",
    author="El Atrach Mohammed Amine",
    description="Automated transcription and scoring pipeline for the Elicited Imitation Task",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
