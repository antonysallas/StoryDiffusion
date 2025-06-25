# /Users/asallasd/workarea/projects/personal/dadboard/application/utils/pip2conda/check_conda_packages.py

import subprocess

from upgrade.logger import logger

def search_conda_package(package_name):
    try:
        # Run conda search command
        result = subprocess.run(
            ["conda", "search", package_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0 and package_name in result.stdout:
            logger.debug(f"Package '{package_name}' found in Conda repository.")
            return True
        else:
            return False
    except Exception as e:
        logger.debug(f"Error searching for package '{package_name}': {e}")
        return False


def install_conda_package(package_name):
    try:
        # Install the package using conda
        subprocess.run(["conda", "install", "-y", package_name], check=True)
        logger.debug(f"Package '{package_name}' installed via Conda.")
    except Exception as e:
        logger.debug(f"Error installing '{package_name}' via Conda: {e}")


def install_pip_package(package_name):
    try:
        # Install the package using pip
        subprocess.run(["pip", "install", package_name], check=True)
        logger.debug(f"Package '{package_name}' installed via Pip.")
    except Exception as e:
        logger.debug(f"Error installing '{package_name}' via Pip: {e}")


def read_requirements(file_path):
    try:
        with open(file_path, "r") as f:
            packages = f.readlines()
        # Filter out lines that start with '#' and empty lines
        packages = [pkg.strip() for pkg in packages if pkg.strip() and not pkg.strip().startswith("#")]
        return [pkg.split("==")[0] for pkg in packages]  # Remove version numbers for conda search
    except Exception as e:
        logger.debug(f"Error reading requirements.txt: {e}")
        return []


def generate_yaml_file(conda_packages, pip_packages, output_file="environment.yaml"):
    try:
        with open(output_file, "w") as f:
            f.write("name: myenv\n")
            f.write("channels:\n")
            f.write("  - defaults\n")
            f.write("dependencies:\n")
            for package in conda_packages:
                f.write(f"  - {package}\n")
            if pip_packages:
                f.write("  - pip:\n")
                for package in pip_packages:
                    f.write(f"      - {package}\n")
        logger.debug(f"Requirements saved to {output_file}")
    except Exception as e:
        logger.debug(f"Error writing to {output_file}: {e}")


def main():
    requirements_file = "requirements.txt"  # Path to your requirements.txt file
    conda_packages = []
    pip_packages = []

    packages = read_requirements(requirements_file)

    logger.debug(f"Processing packages from {requirements_file}...")

    for package in packages:
        if search_conda_package(package):
            install_conda_package(package)
            conda_packages.append(package)
        else:
            logger.debug(f"Package '{package}' not found in Conda repository. Trying pip...")
            install_pip_package(package)
            pip_packages.append(package)

    # Generate environment.yaml
    generate_yaml_file(conda_packages, pip_packages)


if __name__ == "__main__":
    main()
