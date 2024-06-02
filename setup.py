from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
requirements = [line for line in requirements if not line.startswith("#")]

setup(
    name="specdec",
    version="0.1",
    package_dir={"": "specdec"},
    packages=find_packages("specdec"),
    description="Companion code for SpecExec paper and beyond",
    keywords="speculative decoding",
    install_requires=requirements,
    python_requires=">=3.9",
    # If you have scripts that should be directly callable from the command line, you can specify them here.
    scripts=["run_exp.py"],
    package_data={
        # Include any non-code files in your package
        "oasst_prompts": ["data/oasst_prompts.json"],
        "wikitext_prompts": ["data/wikitext_prompts.json"],
    },
    include_package_data=True,
)
