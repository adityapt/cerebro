"""
Package Manager - Detect installed packages and optionally install missing ones
"""
import subprocess
import sys
import logging
from typing import Set, List, Tuple

logger = logging.getLogger(__name__)


class PackageManager:
    """
    Manages Python package detection and installation.
    """
    def __init__(self, auto_install: bool = False, ask_user: bool = True):
        """
        Initialize package manager.
        Args:
            auto_install: If True, automatically install missing packages
            ask_user: If True, ask user before installing (only if auto_install=True)
        """
        self.auto_install = auto_install
        self.ask_user = ask_user
        self._installed_packages = None
        self._essential_packages = {
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy',
            'scikit-learn', 'jax', 'numpyro', 'statsmodels'
        }
    
    def get_installed_packages(self, refresh: bool = False) -> Set[str]:
        """
        Get set of installed package names.
        Args:
            refresh: Force refresh the package list
        Returns:
            Set of lowercase package names
        """
        if self._installed_packages is None or refresh:
            logger.info("  Detecting installed packages...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "list", "--format=freeze"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                packages = set()
                for line in result.stdout.split('\n'):
                    if line.strip() and '==' in line:
                        pkg_name = line.split('==')[0].strip().lower()
                        packages.add(pkg_name)
                self._installed_packages = packages
                logger.info(f"  Found {len(packages)} installed packages")
            except Exception as e:
                logger.error(f"Failed to detect packages: {e}")
                # Return essential packages as fallback
                self._installed_packages = self._essential_packages.copy()
        return self._installed_packages
    
    def check_packages(self, required: List[str]) -> Tuple[List[str], List[str]]:
        """
        Check which packages are installed and which are missing.
        Args:
            required: List of required package names
        Returns:
            (installed, missing) tuple of lists
        """
        installed_set = self.get_installed_packages()
        installed = []
        missing = []
        for pkg in required:
            pkg_lower = pkg.lower().replace('-', '_').replace('_', '-')
            # Check both with hyphens and underscores (scikit-learn vs sklearn)
            pkg_variants = [
                pkg.lower(),
                pkg.lower().replace('-', '_'),
                pkg.lower().replace('_', '-')
            ]
            if any(variant in installed_set for variant in pkg_variants):
                installed.append(pkg)
            else:
                missing.append(pkg)
        return installed, missing
    
    def install_package(self, package_name: str, silent: bool = False) -> bool:
        """
        Install a package using pip.
        Args:
            package_name: Name of package to install
            silent: If True, skip user confirmation (used for auto-install)
        Returns:
            True if installed successfully, False otherwise
        """
        if self.ask_user and not silent:
            response = input(f"\n Install {package_name}? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                logger.info(f"Skipped installing {package_name}")
                return False
        logger.info(f" Installing {package_name}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name, "--quiet"],
                capture_output=True,
                text=True,
                timeout=120
            )
            if result.returncode == 0:
                logger.info(f" ✓ Installed {package_name}")
                # Refresh package list
                self._installed_packages = None
                return True
            else:
                error_msg = result.stderr[:200] if result.stderr else result.stdout[:200]
                logger.error(f" ✗ Failed to install {package_name}: {error_msg}")
                return False
        except Exception as e:
            logger.error(f" ✗ Error installing {package_name}: {e}")
            return False
    def ensure_packages(self, required: List[str]) -> bool:
        """
        Ensure all required packages are installed.
        If auto_install is True, will attempt to install missing packages.
        Args:
            required: List of required package names
        Returns:
            True if all packages are available, False otherwise
        """
        installed, missing = self.check_packages(required)
        if not missing:
            logger.info(f"  All {len(installed)} required packages installed")
            return True
        logger.warning(f"  Missing {len(missing)} packages: {', '.join(missing)}")
        if not self.auto_install:
            logger.info("Set auto_install=True to install automatically")
            return False
        # Try to install missing packages
        all_installed = True
        for pkg in missing:
            if not self.install_package(pkg):
                all_installed = False
        return all_installed
    
    def get_package_constraint_prompt(self) -> str:
        """
        Generate a prompt section that constrains code generation to installed packages.
        Returns:
            Prompt text with available packages
        """
        installed = sorted(self.get_installed_packages())
        # Group by category
        data_packages = [p for p in installed if any(x in p for x in ['pandas', 'numpy', 'scipy', 'statsmodels'])]
        viz_packages = [p for p in installed if any(x in p for x in ['matplotlib', 'seaborn', 'plotly'])]
        ml_packages = [p for p in installed if any(x in p for x in ['sklearn', 'scikit', 'jax', 'numpyro', 'pymc'])]
        prompt = f"""
IMPORTANT: PACKAGE CONSTRAINTS

You must ONLY use packages that are already installed.
Do NOT import packages that are not in this list.

 INSTALLED DATA PACKAGES ({len(data_packages)}):
{', '.join(data_packages[:20])}

 INSTALLED VISUALIZATION PACKAGES ({len(viz_packages)}):
{', '.join(viz_packages[:10])}

 INSTALLED ML/MODELING PACKAGES ({len(ml_packages)}):
{', '.join(ml_packages[:15])}

 Python built-ins are always available:
datetime, collections, itertools, functools, json, logging, warnings, etc.

 DO NOT USE (not installed):
- holidays
- workalendar
- kats (use statsmodels.tsa instead)
- prophet (use statsmodels.tsa.seasonal_decompose instead)
- Any other packages not listed above

INSTEAD: 
- For time series: Use statsmodels.tsa.seasonal_decompose, plot_acf, plot_pacf
- For dates: Use datetime and pandas for date operations

"""
        return prompt


# Global instance (can be configured at startup)
_package_manager = None

def get_package_manager(auto_install: bool = False, ask_user: bool = True) -> PackageManager:
    """Get or create the global package manager instance."""
    global _package_manager
    if _package_manager is None:
        _package_manager = PackageManager(auto_install=auto_install, ask_user=ask_user)
    return _package_manager


def get_package_constraint_prompt() -> str:
    """Convenience function to get package constraint prompt."""
    return get_package_manager().get_package_constraint_prompt()

