from setuptools import find_packages, setup

package_name = "river-policy"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={"": ["py.typed"]},
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="rverma-botcrew",
    maintainer_email="rishabh@botcrew.com",
    description="RL-based cable insertion policy for the AIC qualification phase.",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={"console_scripts": []},
)
