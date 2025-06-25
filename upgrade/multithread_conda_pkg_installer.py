import subprocess
import os
import sys
import json
import re
import argparse
import threading
import time
import re
import sys
import platform

from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from dataclasses import dataclass

# Assuming logger is properly configured in the imported module
from upgrade.logger import logger

# Common name differences between pip and conda
PACKAGE_NAME_MAPPING = {
    "python-dateutil": "dateutil",
    "pillow": "pil",
    "scikit-learn": "sklearn",
    "pytorch": "torch",  # conda uses 'pytorch', pip uses 'torch'
    # Add more mappings as needed
}

IS_MACOS = platform.system() == "Darwin"
IS_APPLE_SILICON = IS_MACOS and platform.machine() == "arm64"

# Create a list of packages known to have issues on Apple Silicon
APPLE_SILICON_PROBLEM_PACKAGES = [
    "xformers",  # Requires complex compilation, often fails
    "triton",    # Common dependency that can cause issues
]

# Additional conda channels to search
DEFAULT_CHANNELS = ["defaults", "conda-forge"]
# DEFAULT_WORKERS = min(32, os.cpu_count() * 4) if os.cpu_count() else 8
DEFAULT_WORKERS = 20


class PackageStatus(Enum):
    PENDING = "pending"
    CHECKING = "checking"
    CONDA_AVAILABLE = "conda_available"
    PIP_ONLY = "pip_only"
    CONDA_INSTALLING = "conda_installing"
    PIP_INSTALLING = "pip_installing"
    CONDA_INSTALLED = "conda_installed"
    PIP_INSTALLED = "pip_installed"
    FAILED = "failed"


@dataclass
class Package:
    name: str
    version_constraint: Optional[str]
    status: PackageStatus = PackageStatus.PENDING
    normalized_name: Optional[str] = None
    error_message: Optional[str] = None


class ThreadSafeCounter:
    def __init__(self, initial=0):
        self.value = initial
        self._lock = threading.Lock()

    def increment(self):
        with self._lock:
            self.value += 1
            return self.value

    def get(self):
        with self._lock:
            return self.value


class CondaPackageManager:
    def __init__(
        self,
        channels: List[str] = DEFAULT_CHANNELS,
        env_name: Optional[str] = None,
        create_env: bool = False,
        requirements_file: str = "requirements.txt",
        output_file: str = "environment.yaml",
        timeout: int = 180,
        workers: int = DEFAULT_WORKERS,
        skip_install: bool = False,
        check_only: bool = False,
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
            workers: Number of worker threads
            skip_install: Skip installation and only check availability
            check_only: Only check package availability, don't install
        """
        self.channels = channels
        self.env_name = env_name
        self.create_env = create_env
        self.requirements_file = requirements_file
        self.output_file = output_file
        self.timeout = timeout
        self.max_workers = workers
        self.skip_install = skip_install
        self.check_only = check_only

        # Check if conda is available
        self.conda_available = self._check_conda_available()
        if not self.conda_available:
            logger.warning("Conda not found. Will use pip for all packages.")

        # Thread-safe storage
        self._lock = threading.Lock()
        self.packages: Dict[str, Package] = {}
        self.conda_packages: List[str] = []
        self.pip_packages: List[str] = []
        self.failed_packages: List[str] = []

        # Cache of available conda packages
        self.available_conda_packages: Set[str] = set()
        self.unavailable_conda_packages: Set[str] = set()
        self.cache_lock = threading.Lock()

        # Progress tracking
        self.progress = {
            "total": 0,
            "checking": ThreadSafeCounter(),
            "installing": ThreadSafeCounter(),
            "completed": ThreadSafeCounter(),
            "failed": ThreadSafeCounter(),
        }

        # Max concurrent subprocesses (to avoid overwhelming the system)
        self.semaphore = threading.Semaphore(min(os.cpu_count() * 2 if os.cpu_count() else 4, 16))

    # Add this function to your CondaPackageManager class to handle platform-specific packages
    def _handle_platform_specific_package(self, package_name: str, version_constraint: Optional[str] = None) -> bool:
        """
        Handle platform-specific package installation.

        Returns:
            True if package was handled, False if it should be processed normally
        """
        if IS_APPLE_SILICON:
            # Special handling for M1/M2 Macs
            if package_name.lower() in [p.lower() for p in APPLE_SILICON_PROBLEM_PACKAGES]:
                logger.warning(f"Package '{package_name}' is known to have issues on Apple Silicon. Skipping installation.")
                return True

            # Special handling for PyTorch on M1/M2
            if package_name.lower() in ["pytorch", "torch"]:
                logger.info(f"Installing PyTorch for Apple Silicon...")
                # Use pip for PyTorch on Apple Silicon as it's more reliable
                if version_constraint:
                    # Keep major.minor version but skip patch version for better compatibility
                    match = re.match(r"==(\d+\.\d+)\.(\d+)", version_constraint)
                    if match:
                        major_minor = match.group(1)
                        cmd = f"pip install --upgrade torch=={major_minor}.* torchvision torchaudio"
                    else:
                        cmd = f"pip install --upgrade torch{version_constraint} torchvision torchaudio"
                else:
                    cmd = "pip install --upgrade torch torchvision torchaudio"

                try:
                    logger.info(f"Running: {cmd}")
                    returncode, stdout, stderr = self._run_command(cmd.split(), check=False)
                    success = returncode == 0

                    if success:
                        logger.info(f"Successfully installed PyTorch for Apple Silicon")
                        with self._lock:
                            self.pip_packages.append("torch")
                            self.pip_packages.append("torchvision")
                            self.pip_packages.append("torchaudio")
                        self.progress["completed"].increment()
                    else:
                        logger.warning(f"Failed to install PyTorch for Apple Silicon: {stderr}")
                        self.progress["failed"].increment()

                    return True
                except Exception as e:
                    logger.error(f"Error installing PyTorch for Apple Silicon: {str(e)}")
                    self.progress["failed"].increment()
                    return True

        return False

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
            # Use semaphore to limit concurrent subprocesses
            with self.semaphore:
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

    def check_conda_availability(self, package: Package) -> bool:
        """
        Check if a package is available in conda.

        Args:
            package: Package to check

        Returns:
            True if package is available in conda, False otherwise
        """
        if not self.conda_available:
            return False

        self.progress["checking"].increment()
        package.status = PackageStatus.CHECKING

        # First check the cache
        with self.cache_lock:
            if package.name.lower() in self.available_conda_packages:
                package.status = PackageStatus.CONDA_AVAILABLE
                return True
            if package.name.lower() in self.unavailable_conda_packages:
                package.status = PackageStatus.PIP_ONLY
                return False

            # If the normalized name is different, check that too
            if package.normalized_name and package.normalized_name.lower() != package.name.lower():
                if package.normalized_name.lower() in self.available_conda_packages:
                    package.status = PackageStatus.CONDA_AVAILABLE
                    return True
                if package.normalized_name.lower() in self.unavailable_conda_packages:
                    package.status = PackageStatus.PIP_ONLY
                    return False

        # Not in cache, need to check with conda
        channel_args = []
        for channel in self.channels:
            channel_args.extend(["-c", channel])

        # Check the original name
        command = ["conda", "search", "--json", package.name] + channel_args
        returncode, stdout, stderr = self._run_command(command, check=False)

        if returncode == 0:
            try:
                data = json.loads(stdout)
                if data:
                    with self.cache_lock:
                        self.available_conda_packages.add(package.name.lower())
                    package.status = PackageStatus.CONDA_AVAILABLE
                    return True
                else:
                    with self.cache_lock:
                        self.unavailable_conda_packages.add(package.name.lower())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response for {package.name}")

        # If normalized name is different, check that too
        if package.normalized_name and package.normalized_name.lower() != package.name.lower():
            command = ["conda", "search", "--json", package.normalized_name] + channel_args
            returncode, stdout, stderr = self._run_command(command, check=False)

            if returncode == 0:
                try:
                    data = json.loads(stdout)
                    if data:
                        with self.cache_lock:
                            self.available_conda_packages.add(package.normalized_name.lower())
                        package.status = PackageStatus.CONDA_AVAILABLE
                        return True
                    else:
                        with self.cache_lock:
                            self.unavailable_conda_packages.add(package.normalized_name.lower())
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON response for {package.normalized_name}")

        # Not available in conda
        package.status = PackageStatus.PIP_ONLY
        return False

    def install_conda_package(self, package: Package) -> bool:
        """
        Install a package using conda.

        Args:
            package: Package to install

        Returns:
            True if installation was successful, False otherwise
        """
        if not self.conda_available or self.skip_install or self.check_only:
            return False

        package.status = PackageStatus.CONDA_INSTALLING
        self.progress["installing"].increment()

        # Prepare the package specification
        package_spec = package.normalized_name or package.name
        if package.version_constraint:
            # Convert pip-style version constraints to conda-style
            if package.version_constraint.startswith("=="):
                package_spec = f"{package_spec}={package.version_constraint[2:]}"
            elif package.version_constraint.startswith(">="):
                package_spec = f"{package_spec}>={package.version_constraint[2:]}"
            elif package.version_constraint.startswith("<="):
                package_spec = f"{package_spec}<={package.version_constraint[2:]}"
            elif package.version_constraint.startswith(">"):
                package_spec = f"{package_spec}>{package.version_constraint[1:]}"
            elif package.version_constraint.startswith("<"):
                package_spec = f"{package_spec}<{package.version_constraint[1:]}"
            else:
                # Other constraints like '~=' aren't directly supported in conda
                package_spec = package.normalized_name or package.name

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
                with self._lock:
                    self.conda_packages.append(package_spec)
                package.status = PackageStatus.CONDA_INSTALLED
                self.progress["completed"].increment()
            else:
                logger.warning(f"Failed to install {package_spec} via conda: {stderr}")
                package.error_message = f"Conda install failed: {stderr}"

            return success
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error installing {package_spec} via conda: {error_msg}")
            package.error_message = f"Conda install error: {error_msg}"
            return False

    def install_pip_package(self, package: Package, override_name: Optional[str] = None) -> bool:
        """
        Install a package using pip.

        Args:
            package: Package to install
            override_name: Override the package name (for handling name differences)

        Returns:
            True if installation was successful, False otherwise
        """
        if self.skip_install or self.check_only:
            return False

        package.status = PackageStatus.PIP_INSTALLING
        self.progress["installing"].increment()

        # Use the override name if provided
        package_name = override_name or package.name

        # Prepare the package specification
        package_spec = package_name
        if package.version_constraint:
            package_spec = f"{package_name}{package.version_constraint}"

        # Prepare the command
        command = ["pip", "install", package_spec]

        logger.info(f"Installing {package_spec} via pip...")
        try:
            returncode, stdout, stderr = self._run_command(command, check=False)
            success = returncode == 0

            if success:
                logger.info(f"Successfully installed {package_spec} via pip")
                with self._lock:
                    self.pip_packages.append(package_spec)
                package.status = PackageStatus.PIP_INSTALLED
                self.progress["completed"].increment()
            else:
                logger.warning(f"Failed to install {package_spec} via pip: {stderr}")
                with self._lock:
                    self.failed_packages.append(package_spec)
                package.status = PackageStatus.FAILED
                package.error_message = f"Pip install failed: {stderr}"
                self.progress["failed"].increment()

            return success
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error installing {package_spec} via pip: {error_msg}")
            with self._lock:
                self.failed_packages.append(package_spec)
            package.status = PackageStatus.FAILED
            package.error_message = f"Pip install error: {error_msg}"
            self.progress["failed"].increment()
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

    def _process_package(self, package: Package) -> None:
        """Process a single package (check and install)."""
        # This must be idempotent and thread-safe

        if package.status == PackageStatus.PENDING:
            # First normalize the package name
            package.normalized_name = self._normalize_package_name(package.name)

            # Handle platform-specific packages (e.g., Apple Silicon)
            if self._handle_platform_specific_package(package.name, package.version_constraint):
                package.status = PackageStatus.COMPLETED
                return

            # Check if available in conda
            conda_available = self.check_conda_availability(package)

            # Skip installation if only checking
            if self.skip_install or self.check_only:
                self.progress["completed"].increment()
                return

            # Install with conda if available
            if conda_available:
                if self.install_conda_package(package):
                    return
                logger.warning(f"Falling back to pip for {package.name}")

            # Try pip installation if conda failed or not available
            # For PyPI, we need to use the correct package name
            pip_package_name = package.name
            if package.name.lower() == "pytorch" and package.normalized_name.lower() == "torch":
                pip_package_name = "torch"  # Use 'torch' for pip instead of 'pytorch'

            if not self.install_pip_package(package, override_name=pip_package_name):
                logger.error(f"Failed to install {package.name}")
                package.status = PackageStatus.FAILED
                self.progress["failed"].increment()

    def _print_progress(self, start_time: float, interval: int = 2) -> None:
        """Print progress updates at regular intervals."""
        while True:
            time.sleep(interval)
            elapsed = time.time() - start_time
            total = self.progress["total"]
            checking = self.progress["checking"].get()
            installing = self.progress["installing"].get()
            completed = self.progress["completed"].get()
            failed = self.progress["failed"].get()

            if completed + failed >= total:
                break

            logger.info(
                f"Progress: {completed}/{total} completed, {failed} failed, "
                f"{checking} checked, {installing} installed. "
                f"Elapsed: {elapsed:.1f}s"
            )

    def process_packages(self) -> bool:
        """
        Process all packages from requirements file.

        Returns:
            True if all packages were successfully installed, False otherwise
        """
        start_time = time.time()

        # Create environment if needed
        if self.create_env and not self.create_conda_environment():
            logger.error("Failed to create conda environment. Aborting.")
            return False

        # Read requirements
        package_specs = self.read_requirements()
        if not package_specs:
            logger.error(f"No packages found in {self.requirements_file}")
            return False

        # Create package objects and add to dict
        for name, version in package_specs:
            self.packages[name] = Package(name=name, version_constraint=version)

        self.progress["total"] = len(package_specs)
        logger.info(f"Processing {len(package_specs)} packages...")

        # Start progress reporting thread
        progress_thread = threading.Thread(
            target=self._print_progress,
            args=(start_time,),
            daemon=True
        )
        progress_thread.start()

        # Process packages in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all packages for processing
            futures = {
                executor.submit(self._process_package, package): name
                for name, package in self.packages.items()
            }

            # Process results as they complete
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()  # This will propagate any exceptions
                except Exception as e:
                    logger.error(f"Error processing package {name}: {str(e)}")
                    self.packages[name].status = PackageStatus.FAILED
                    self.packages[name].error_message = str(e)
                    self.progress["failed"].increment()

        # Wait for progress thread to finish
        progress_thread.join()

        # Generate YAML file
        if not self.check_only:
            self.generate_yaml_file()

        # Report results
        elapsed = time.time() - start_time
        total = len(package_specs)
        conda_count = len(self.conda_packages)
        pip_count = len(self.pip_packages)
        failed_count = len(self.failed_packages)

        logger.info(f"Package installation summary (took {elapsed:.1f}s):")
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
        default=180,
        help="Timeout for conda commands in seconds"
    )

    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of worker threads"
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check package availability, don't install"
    )

    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installation and only check availability"
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
        timeout=args.timeout,
        workers=args.workers,
        skip_install=args.skip_install,
        check_only=args.check_only
    )

    success = manager.process_packages()

    if not success:
        logger.warning("Some packages failed to install. Check logs for details.")
        sys.exit(1)

    logger.info("Successfully processed all packages.")


if __name__ == "__main__":
    main()