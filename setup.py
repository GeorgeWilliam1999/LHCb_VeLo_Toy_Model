from setuptools import setup, find_packages

setup(
    name="LHCB_Velo_Toy_Models",  # Change to your project name
    version="0.1.0",
    author="George William Scriven",  # Replace with your name
    author_email="george.w.scriven@gmail.com",
    description="A simulation framework for the LHCb VELO detector",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GeorgeWilliam1999/LHCb_VeLo_Toy_Model",  # Replace with your repo URL
    # packages=find_packages(where=["LHCB_Velo_Toy_Models/detector_geometries",
    #                               "LHCB_Velo_Toy_Models/generators",
    #                               "LHCB_Velo_Toy_Models/state_event_generator",
    #                               "LHCB_Velo_Toy_Models/state_event_model"]),  # Automatically find submodules
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Set minimum Python version
)
