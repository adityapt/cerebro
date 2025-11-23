"""
Process manager to prevent multiple model instances from running simultaneously.
Critical for preventing system crashes due to RAM exhaustion.
"""
import os
import signal
import subprocess
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def kill_existing_mlx_processes():
    """
    Kill any existing MLX/model processes to free up RAM.
    This prevents multiple large models from loading simultaneously.
    """
    try:
        # Find Python processes running MLX or large models
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        lines = result.stdout.split('\n')
        killed_count = 0
        
        # Look for processes to kill (but NOT current pipeline runs)
        keywords = ['ollama serve']  # Only kill Ollama server, not Python scripts
        current_pid = os.getpid()
        parent_pid = os.getppid()
        
        for line in lines:
            # Skip header and current process
            if 'PID' in line or str(current_pid) in line:
                continue
            
            # Check if line contains any keywords
            if any(kw in line.lower() for kw in keywords):
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                try:
                    pid = int(parts[1])
                    process_name = ' '.join(parts[10:])[:50]
                    
                    # Don't kill the current process
                    if pid == current_pid:
                        continue
                    
                    logger.info(f"🔪 Killing process {pid}: {process_name}")
                    os.kill(pid, signal.SIGTERM)
                    killed_count += 1
                    
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    continue
        
        if killed_count > 0:
            logger.info(f"✓ Killed {killed_count} existing process(es)")
            # Give processes time to clean up
            time.sleep(2)
        else:
            logger.info("✓ No existing processes found")
        
        return killed_count
        
    except subprocess.TimeoutExpired:
        logger.warning("Process check timed out")
        return 0
    except Exception as e:
        logger.warning(f"Could not check/kill processes: {e}")
        return 0


def ensure_single_instance(lockfile_path: str = None):
    """
    Ensure only one instance of the pipeline is running.
    Uses a lockfile to prevent concurrent runs.
    
    Args:
        lockfile_path: Path to lockfile (default: /tmp/cerebro_pipeline.lock)
    """
    if lockfile_path is None:
        lockfile_path = "/tmp/cerebro_pipeline.lock"
    
    lockfile = Path(lockfile_path)
    
    # Check if lockfile exists
    if lockfile.exists():
        try:
            # Read PID from lockfile
            with open(lockfile, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if that process is still running
            try:
                os.kill(old_pid, 0)  # Signal 0 checks if process exists
                logger.warning(f"⚠️  Another pipeline instance is running (PID: {old_pid})")
                logger.warning("Killing it to prevent memory exhaustion...")
                os.kill(old_pid, signal.SIGTERM)
                time.sleep(2)
            except ProcessLookupError:
                # Process doesn't exist, safe to proceed
                pass
                
        except Exception as e:
            logger.warning(f"Could not read lockfile: {e}")
    
    # Write current PID to lockfile
    try:
        with open(lockfile, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"✓ Lockfile created: {lockfile}")
    except Exception as e:
        logger.warning(f"Could not create lockfile: {e}")


def cleanup_on_exit(lockfile_path: str = None):
    """
    Clean up lockfile on exit.
    Should be called in finally block or atexit handler.
    """
    if lockfile_path is None:
        lockfile_path = "/tmp/cerebro_pipeline.lock"
    
    lockfile = Path(lockfile_path)
    
    if lockfile.exists():
        try:
            lockfile.unlink()
            logger.info("✓ Lockfile cleaned up")
        except Exception as e:
            logger.warning(f"Could not remove lockfile: {e}")


def check_system_resources():
    """
    Check if system has enough resources to run the pipeline.
    Returns dict with system status.
    """
    try:
        # Get memory info on macOS
        result = subprocess.run(
            ['vm_stat'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        # Parse output
        lines = result.stdout.split('\n')
        free_pages = 0
        
        for line in lines:
            if 'Pages free:' in line:
                free_pages = int(line.split(':')[1].strip().replace('.', ''))
                break
        
        # Each page is 16KB on macOS
        free_gb = (free_pages * 16384) / (1024**3)
        
        # Check if we have at least 25GB free (for 32B model)
        has_enough_ram = free_gb >= 25
        
        return {
            'free_gb': free_gb,
            'has_enough_ram': has_enough_ram,
            'recommended_model': '32b' if free_gb >= 25 else '7b'
        }
        
    except Exception as e:
        logger.warning(f"Could not check system resources: {e}")
        return {
            'free_gb': None,
            'has_enough_ram': True,  # Assume okay if check fails
            'recommended_model': '7b'  # Default to smaller model
        }

