#!/usr/bin/env python
"""
Extract package information from a conda environment and create
outputs in various formats (YAML, requirements.txt, etc.)
"""

import os
import sys
import json
import subprocess
import argparse

def get_conda_env_list():
    """Get list of available conda environments."""
    result = subprocess.run(
        ["conda", "env", "list", "--json"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        print(f"Error getting environment list: {result.stderr}")
        return {}

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing conda environment list: {result.stdout}")
        return {}

def get_environment_packages(env_name=None, env_path=None):
    """Get packages installed in the specified environment."""
    if env_name:
        cmd = ["conda", "list", "--name", env_name, "--json"]
    elif env_path:
        cmd = ["conda", "list", "--prefix", env_path, "--json"]
    else:
        # Use current environment
        cmd = ["conda", "list", "--json"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting package list: {result.stderr}")
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing package list: {result.stdout}")
        return []

def get_pip_packages(env_name=None, env_path=None):
    """Get packages installed via pip."""
    if env_name:
        # Get the environment path from the name
        envs = get_conda_env_list()
        if 'envs' not in envs:
            print(f"Could not find environment list")
            return []

        for env in envs['envs']:
            if env.endswith(os.path.sep + env_name):
                env_path = env
                break

        if not env_path:
            print(f"Could not find environment '{env_name}'")
            return []

    if env_path:
        # Construct the path to pip within this environment
        if sys.platform == 'win32':
            pip_path = os.path.join(env_path, "Scripts", "pip")
        else:
            pip_path = os.path.join(env_path, "bin", "pip")

        cmd = [pip_path, "list", "--format=json"]
    else:
        # Use current environment's pip
        cmd = ["pip", "list", "--format=json"]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error getting pip package list: {result.stderr}")
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"Error parsing pip package list: {result.stdout}")
        return []

def write_requirements_txt(packages, output_file="requirements.txt"):
    """Write packages to a requirements.txt file."""
    with open(output_file, "w") as f:
        for package in sorted(packages, key=lambda p: p['name'].lower()):
            if package['name'].lower() != 'python':  # Skip Python itself
                f.write(f"{package['name']}=={package['version']}\n")

def write_environment_yaml(conda_packages, pip_packages, output_file="environment.yaml", env_name="myenv", python_version=None):
    """Write packages to an environment.yaml file."""
    with open(output_file, "w") as f:
        f.write(f"name: {env_name}\n")
        f.write("channels:\n")
        f.write("  - conda-forge\n")
        f.write("  - defaults\n")
        f.write("dependencies:\n")

        # Add Python version if specified
        if python_version:
            f.write(f"  - python={python_version}\n")

        # Add conda packages
        for package in sorted(conda_packages, key=lambda p: p['name'].lower()):
            if package['name'].lower() != 'python':  # Skip Python as it's already included
                f.write(f"  - {package['name']}=={package['version']}\n")

        # Add pip packages if any
        if pip_packages:
            f.write("  - pip\n")
            f.write("  - pip:\n")
            for package in sorted(pip_packages, key=lambda p: p['name'].lower()):
                f.write(f"    - {package['name']}=={package['version']}\n")

def main():
    parser = argparse.ArgumentParser(description="Extract package information from a conda environment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--name", "-n", help="Name of conda environment")
    group.add_argument("--path", "-p", help="Path to conda environment")

    parser.add_argument("--output-dir", "-o", default=".", help="Directory to save output files")
    parser.add_argument("--python-version", help="Python version for new environment")
    parser.add_argument("--requirements", action="store_true", help="Generate requirements.txt")
    parser.add_argument("--conda-yaml", action="store_true", help="Generate environment.yaml for conda")
    parser.add_argument("--all", "-a", action="store_true", help="Generate all output formats")
    parser.add_argument("--target-env-name", default="myenv", help="Environment name for YAML file")

    args = parser.parse_args()

    # Default to generating all outputs if none specified
    if not any([args.requirements, args.conda_yaml, args.all]):
        args.all = True

    # Get packages from conda
    conda_packages = get_environment_packages(args.name, args.path)

    # Get packages from pip
    pip_packages = get_pip_packages(args.name, args.path)

    # Convert pip packages to the same format as conda packages
    formatted_pip_packages = [
        {"name": pkg["name"], "version": pkg["version"], "channel": "pip"}
        for pkg in pip_packages
    ]

    # Get Python version from packages
    python_version = args.python_version
    if not python_version:
        for pkg in conda_packages:
            if pkg['name'].lower() == 'python':
                python_version = pkg['version']
                break

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate requested outputs
    if args.all or args.requirements:
        # Combine conda and pip packages
        all_packages = conda_packages + formatted_pip_packages
        write_requirements_txt(
            all_packages,
            os.path.join(args.output_dir, "requirement_" + python_version + ".txt")
        )
        print(f"Generated requirements.txt")

    if args.all or args.conda_yaml:
        # Separate conda and pip packages
        conda_only = [pkg for pkg in conda_packages if pkg['channel'] != 'pypi']
        write_environment_yaml(
            conda_only,
            formatted_pip_packages,
            os.path.join(args.output_dir, "environment.yaml"),
            args.target_env_name,
            python_version
        )
        print(f"Generated environment.yaml")

    # Print statistics
    conda_count = len([pkg for pkg in conda_packages if pkg['channel'] != 'pypi'])
    pip_count = len(formatted_pip_packages)

    print(f"\nEnvironment statistics:")
    print(f"  Total packages: {conda_count + pip_count}")
    print(f"  Conda packages: {conda_count}")
    print(f"  Pip packages: {pip_count}")
    print(f"  Python version: {python_version}")

if __name__ == "__main__":
    main()