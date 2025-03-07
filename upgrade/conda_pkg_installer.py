import subprocess
import sys
import json
import re
import argparse
from typing import List, Tuple, Optional, Set

# Assuming logger is properly configured in the imported module
from upgrade.logger import logger

# Common name differences between pip and conda
PACKAGE_NAME_MAPPING = {
    "python-dateutil": "dateutil",
    "pillow": "pil",
    "scikit-learn": "sklearn",
    # Add more mappings as needed
}

# Additional conda channels to search
DEFAULT_CHANNELS = ["defaults", "conda-forge"]


class CondaPackageManager:
    def __init__(
        self,
        channels: List[str] = DEFAULT_CHANNELS,
        env_name: Optional[str] = None,
        create_env: bool = False,
        requirements_file: str = "requirements.txt",
        output_file: str = "environment.yaml",
        timeout: int = 300,
    ):
        """
        Initialize the conda package manager.

        Args:
            channels: List of conda channels to search
            env_name: Name of conda environment to use
            create_env: Whether to create a new environment
            requirements_file: Path to requirements.txt file
            output_file: Path to output environment.yaml file
            timeout: Timeout for conda commands in seconds
        """
        self.channels = channels
        self.env_name = env_name
        self.create_env = create_env
        self.requirements_file = requirements_file
        self.output_file = output_file
        self.timeout = timeout

        # Check if conda is available
        self.conda_available = self._check_conda_available()
        if not self.conda_available:
            logger.warning("Conda not found. Will use pip for all packages.")

        # Storage for processed packages
        self.conda_packages: List[str] = []
        self.pip_packages: List[str] = []
        self.failed_packages: List[str] = []

        # Cache the conda package list to avoid repeated searches
        self.available_conda_packages: Set[str] = set()

    def _check_conda_available(self) -> bool:
        """Check if conda is available in the system."""
        try:
            result = subprocess.run(
                ["conda", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def _run_command(self, command: List[str], check: bool = True) -> Tuple[int, str, str]:
        """
        Run a shell command and return the result.

        Args:
            command: Command to run as a list of strings
            check: Whether to check return code

        Returns:
            Tuple of (return_code, stdout, stderr)

        Raises:
            subprocess.CalledProcessError: If check is True and command fails
        """
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=check,
                timeout=self.timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {self.timeout} seconds: {' '.join(command)}")
            return 1, "", f"Timeout after {self.timeout} seconds"
        except subprocess.CalledProcessError as e:
            if check:
                raise
            return e.returncode, e.stdout, e.stderr
        except Exception as e:
            logger.error(f"Error running command {' '.join(command)}: {str(e)}")
            if check:
                raise
            return 1, "", str(e)

    def _normalize_package_name(self, package_name: str) -> str:
        """
        Normalize package name for conda search.

        Args:
            package_name: Package name to normalize

        Returns:
            Normalized package name
        """
        # Remove any extras (e.g., package[extra])
        package_name = re.sub(r'\[.*\]', '', package_name)

        # Check if package name needs to be mapped
        for pip_name, conda_name in PACKAGE_NAME_MAPPING.items():
            if package_name.lower() == pip_name.lower():
                return conda_name

        return package_name

    def _parse_package_spec(self, package_spec: str) -> Tuple[str, Optional[str]]:
        """
        Parse package specification to get name and version.

        Args:
            package_spec: Package specification (e.g., 'numpy==1.19.5')

        Returns:
            Tuple of (package_name, version_constraint)
        """
        # Handle requirement with version spec
        if '==' in package_spec:
            name, version = package_spec.split('==', 1)
            return name, f"=={version}"
        elif '>=' in package_spec:
            name, version = package_spec.split('>=', 1)
            return name, f">={version}"
        elif '<=' in package_spec:
            name, version = package_spec.split('<=', 1)
            return name, f"<={version}"
        elif '>' in package_spec:
            name, version = package_spec.split('>', 1)
            return name, f">{version}"
        elif '<' in package_spec:
            name, version = package_spec.split('<', 1)
            return name, f"<{version}"
        elif '~=' in package_spec:
            name, version = package_spec.split('~=', 1)
            return name, f"~={version}"
        elif '!=' in package_spec:
            name, version = package_spec.split('!=', 1)
            return name, f"!={version}"
        # Handle package with extras but no version
        elif '[' in package_spec:
            match = re.match(r'([^\[]+)(\[.+\])', package_spec)
            if match:
                return match.group(1), match.group(2)

        # No version constraint
        return package_spec, None

    def read_requirements(self) -> List[Tuple[str, Optional[str]]]:
        """
        Read requirements from file and parse package names and versions.

        Returns:
            List of tuples (package_name, version_constraint)
        """
        try:
            with open(self.requirements_file, "r") as f:
                packages = f.readlines()

            # Filter out comments and empty lines
            packages = [pkg.strip() for pkg in packages if pkg.strip() and not pkg.strip().startswith("#")]

            # Parse package specifications
            return [self._parse_package_spec(pkg) for pkg in packages]
        except Exception as e:
            logger.error(f"Error reading requirements file {self.requirements_file}: {e}")
            return []

    def _populate_conda_package_cache(self):
        """Populate cache of available conda packages from all channels."""
        if not self.conda_available:
            return

        logger.info("Building conda package cache...")

        # Since the --all flag is not supported in all conda versions,
        # we'll use a simpler approach with a list of common packages
        common_packages = [
            "numpy", "pandas", "matplotlib", "scipy", "scikit-learn",
            "torch", "tensorflow", "keras", "flask", "django", "requests",
            "beautifulsoup4", "pillow", "opencv", "nltk", "spacy",
            "jupyter", "pytest", "tqdm", "seaborn", "plotly"
        ]

        for package in common_packages:
            self._check_package_availability(package)

        logger.info(f"Conda package cache contains {len(self.available_conda_packages)} packages")

    def _check_package_availability(self, package_name: str) -> bool:
        """
        Check if a package is available in conda and add it to the cache if it is.

        Args:
            package_name: Package name to check

        Returns:
            True if package is available in conda, False otherwise
        """
        channel_args = []
        for channel in self.channels:
            channel_args.extend(["-c", channel])

        command = ["conda", "search", "--json", package_name] + channel_args
        logger.debug(f"Checking availability of {package_name} in conda...")

        returncode, stdout, stderr = self._run_command(command, check=False)

        if returncode == 0:
            logger.debug(f"Package '{package_name}' found in Conda repository.")
            try:
                data = json.loads(stdout)
                if data:
                    self.available_conda_packages.add(package_name.lower())
                    return True
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for {package_name}")

        return False

    def is_available_in_conda(self, package_name: str) -> bool:
        """
        Check if a package is available in conda.

        Args:
            package_name: Package name to check

        Returns:
            True if package is available in conda, False otherwise
        """
        if not self.conda_available:
            return False

        # Try the original package name
        if package_name.lower() in self.available_conda_packages:
            return True

        # Try the normalized package name
        normalized_name = self._normalize_package_name(package_name)
        if normalized_name.lower() in self.available_conda_packages:
            return True

        # Direct search for the package
        if self._check_package_availability(package_name):
            return True

        # Also try with the normalized name if different
        if normalized_name != package_name and self._check_package_availability(normalized_name):
            return True

        return False

    def install_conda_package(self, package_name: str, version_constraint: Optional[str] = None) -> bool:
        """
        Install a package using conda.

        Args:
            package_name: Package name to install
            version_constraint: Version constraint (e.g., '==1.19.5')

        Returns:
            True if installation was successful, False otherwise
        """
        if not self.conda_available:
            return False

        # Prepare the package specification
        package_spec = package_name
        if version_constraint:
            # Convert pip-style version constraints to conda-style
            if version_constraint.startswith("=="):
                package_spec = f"{package_name}={version_constraint[2:]}"
            elif version_constraint.startswith(">="):
                package_spec = f"{package_name}>={version_constraint[2:]}"
            elif version_constraint.startswith("<="):
                package_spec = f"{package_name}<={version_constraint[2:]}"
            elif version_constraint.startswith(">"):
                package_spec = f"{package_name}>{version_constraint[1:]}"
            elif version_constraint.startswith("<"):
                package_spec = f"{package_name}<{version_constraint[1:]}"
            else:
                # Other constraints like '~=' aren't directly supported in conda
                package_spec = package_name

        # Prepare the command
        channel_args = []
        for channel in self.channels:
            channel_args.extend(["-c", channel])

        command = ["conda", "install", "-y"] + channel_args + [package_spec]

        # Add environment name if specified
        if self.env_name:
            command.extend(["-n", self.env_name])

        logger.info(f"Installing {package_spec} via conda...")
        try:
            returncode, stdout, stderr = self._run_command(command, check=False)
            success = returncode == 0

            if success:
                logger.info(f"Successfully installed {package_spec} via conda")
                self.conda_packages.append(package_spec)
            else:
                logger.warning(f"Failed to install {package_spec} via conda: {stderr}")

            return success
        except Exception as e:
            logger.error(f"Error installing {package_spec} via conda: {str(e)}")
            return False

    def install_pip_package(self, package_name: str, version_constraint: Optional[str] = None) -> bool:
        """
        Install a package using pip.

        Args:
            package_name: Package name to install
            version_constraint: Version constraint (e.g., '==1.19.5')

        Returns:
            True if installation was successful, False otherwise
        """
        # Prepare the package specification
        package_spec = package_name
        if version_constraint:
            package_spec = f"{package_name}{version_constraint}"

        # Prepare the command
        command = ["pip", "install", package_spec]

        logger.info(f"Installing {package_spec} via pip...")
        try:
            returncode, stdout, stderr = self._run_command(command, check=False)
            success = returncode == 0

            if success:
                logger.info(f"Successfully installed {package_spec} via pip")
                self.pip_packages.append(package_spec)
            else:
                logger.warning(f"Failed to install {package_spec} via pip: {stderr}")
                self.failed_packages.append(package_spec)

            return success
        except Exception as e:
            logger.error(f"Error installing {package_spec} via pip: {str(e)}")
            self.failed_packages.append(package_spec)
            return False

    def create_conda_environment(self) -> bool:
        """
        Create a new conda environment if needed.

        Returns:
            True if creation was successful or not needed, False otherwise
        """
        if not self.conda_available or not self.create_env or not self.env_name:
            return True

        logger.info(f"Creating conda environment '{self.env_name}'...")
        command = ["conda", "create", "-y", "-n", self.env_name, "python"]

        try:
            returncode, stdout, stderr = self._run_command(command, check=False)
            success = returncode == 0

            if success:
                logger.info(f"Created conda environment '{self.env_name}'")
            else:
                logger.error(f"Failed to create conda environment '{self.env_name}': {stderr}")

            return success
        except Exception as e:
            logger.error(f"Error creating conda environment '{self.env_name}': {str(e)}")
            return False

    def generate_yaml_file(self) -> bool:
        """
        Generate an environment.yaml file from installed packages.

        Returns:
            True if file was successfully written, False otherwise
        """
        try:
            with open(self.output_file, "w") as f:
                f.write(f"name: {self.env_name or 'myenv'}\n")
                f.write("channels:\n")
                for channel in self.channels:
                    f.write(f"  - {channel}\n")
                f.write("dependencies:\n")
                f.write("  - python\n")

                for package in self.conda_packages:
                    f.write(f"  - {package}\n")

                if self.pip_packages:
                    f.write("  - pip\n")
                    f.write("  - pip:\n")
                    for package in self.pip_packages:
                        f.write(f"    - {package}\n")

            logger.info(f"Generated environment file at {self.output_file}")
            return True
        except Exception as e:
            logger.error(f"Error generating environment file {self.output_file}: {str(e)}")
            return False

    def process_packages(self) -> bool:
        """
        Process all packages from requirements file.

        Returns:
            True if all packages were successfully installed, False otherwise
        """
        # Create environment if needed
        if self.create_env and not self.create_conda_environment():
            logger.error("Failed to create conda environment. Aborting.")
            return False

        # Read requirements
        package_specs = self.read_requirements()
        if not package_specs:
            logger.error(f"No packages found in {self.requirements_file}")
            return False

        logger.info(f"Processing {len(package_specs)} packages...")

        # Initialize available_conda_packages with common packages to improve performance
        # This is now a lighter implementation that doesn't use the problematic '--all' flag
        if self.conda_available and not self.available_conda_packages:
            self._populate_conda_package_cache()

        # Process each package
        for package_name, version_constraint in package_specs:
            normalized_name = self._normalize_package_name(package_name)

            if self.conda_available and self.is_available_in_conda(normalized_name):
                if not self.install_conda_package(normalized_name, version_constraint):
                    logger.warning(f"Falling back to pip for {package_name}")
                    if not self.install_pip_package(package_name, version_constraint):
                        logger.error(f"Failed to install {package_name} with both conda and pip")
            else:
                logger.info(f"Package {package_name} not found in conda, installing with pip")
                if not self.install_pip_package(package_name, version_constraint):
                    logger.error(f"Failed to install {package_name} via pip")

        # Generate YAML file
        self.generate_yaml_file()

        # Report results
        total = len(package_specs)
        conda_count = len(self.conda_packages)
        pip_count = len(self.pip_packages)
        failed_count = len(self.failed_packages)

        logger.info(f"Package installation summary:")
        logger.info(f"  Total packages: {total}")
        logger.info(f"  Installed with conda: {conda_count}")
        logger.info(f"  Installed with pip: {pip_count}")
        logger.info(f"  Failed installations: {failed_count}")

        if failed_count > 0:
            logger.warning("Failed packages:")
            for package in self.failed_packages:
                logger.warning(f"  - {package}")
            return False

        return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Install packages using conda with pip fallback")

    parser.add_argument(
        "-f", "--requirements-file",
        default="requirements.txt",
        help="Path to requirements.txt file"
    )

    parser.add_argument(
        "-o", "--output-file",
        default="environment.yaml",
        help="Path to output environment.yaml file"
    )

    parser.add_argument(
        "-e", "--environment",
        help="Conda environment name to use"
    )

    parser.add_argument(
        "--create-env",
        action="store_true",
        help="Create a new conda environment if it doesn't exist"
    )

    parser.add_argument(
        "-c", "--channels",
        nargs="+",
        default=DEFAULT_CHANNELS,
        help="Conda channels to search for packages"
    )

    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Timeout for conda commands in seconds"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    manager = CondaPackageManager(
        channels=args.channels,
        env_name=args.environment,
        create_env=args.create_env,
        requirements_file=args.requirements_file,
        output_file=args.output_file,
        timeout=args.timeout
    )

    success = manager.process_packages()

    if not success:
        logger.warning("Some packages failed to install. Check logs for details.")
        sys.exit(1)

    logger.info("Successfully installed all packages.")


if __name__ == "__main__":
    main()