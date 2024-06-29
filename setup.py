import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = list()
    for req in open("requirements.txt").readlines():
        req = req.strip()
        if "git+" in req and "util" in req:
            req = f"util-colleenjg @ {req}"
        requirements.append(req)

setuptools.setup(
    name="credassign",
    version="0.0.1",
    author="Colleen J. Gillon",
    author_email="colleen.gillon@utoronto.ca",
    description="Package for analysis of OpenScope Credit Assignment data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/colleenjg/OpenScope_CA_Analysis/tree/minimal",
    packages=setuptools.find_packages(),
    include_package_data=True,
    entry_points={"console_scripts": ["run_paper_figures=credassign.run_paper_figures:main"]},
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
