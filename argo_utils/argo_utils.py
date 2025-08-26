#!/usr/bin/env python3
"""
Argo Utilities

Consolidated utilities for managing Argo proxy and configuration:
- Automatically manages local Argo proxy service
- Detects Argo configurations and starts proxy when needed
- Provides status information and validation
"""

import os
import socket
import subprocess
import time
import threading
import signal
import sys
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ArgoStatus:
    """Status information for Argo proxy"""
    is_running: bool
    host: str
    port: int
    executable_found: bool
    managed_by_us: bool
    config_file: Optional[str] = None
    is_argo_config: bool = False


class ArgoManager:
    """Manages Argo proxy and configuration automatically"""
    
    def __init__(self, host: str = "localhost", port: int = 61045, timeout: int = 2):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.proxy_process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        
    def is_proxy_running(self) -> bool:
        """Check if the Argo proxy is running on the specified host and port"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            result = sock.connect_ex((self.host, self.port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    def find_argo_proxy_executable(self) -> Optional[str]:
        """Find the argo-proxy executable"""
        # Check if argo-proxy is installed and available
        try:
            result = subprocess.run(["which", "argo-proxy"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        # Also check if it's available via pip
        try:
            result = subprocess.run([sys.executable, "-m", "argo_proxy", "--help"], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return f"{sys.executable} -m argo_proxy"
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass
        
        return None
    
    def start_proxy(self) -> bool:
        """Start the Argo proxy if it's not already running"""
        with self._lock:
            if self.is_proxy_running():
                return True
            
            if self.proxy_process and self.proxy_process.poll() is None:
                return True
            
            executable = self.find_argo_proxy_executable()
            if not executable:
                print(f"Error: argo-proxy not found")
                print("Please install it with: pip install argo-proxy")
                print("Or ensure it's available in your PATH")
                return False
            
            try:
                print(f"Starting Argo proxy service...")
                print(f"Proxy will be available at: http://{self.host}:{self.port}")
                
                # Start the argo-proxy service
                if executable.startswith(f"{sys.executable} -m"):
                    # Run via python module
                    cmd = executable.split()
                    self.proxy_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
                else:
                    # Run direct executable
                    self.proxy_process = subprocess.Popen(
                        [executable],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid if hasattr(os, 'setsid') else None
                    )
                
                # Wait a bit for the proxy to start
                time.sleep(2)
                
                # Check if it's now running
                if self.is_proxy_running():
                    print("✓ Argo proxy started successfully")
                    return True
                else:
                    print("✗ Argo proxy failed to start")
                    if self.proxy_process:
                        self.proxy_process.terminate()
                        self.proxy_process = None
                    return False
                    
            except Exception as e:
                print(f"Error starting Argo proxy: {e}")
                if self.proxy_process:
                    self.proxy_process.terminate()
                    self.proxy_process = None
                return False
    
    def ensure_proxy_running(self) -> bool:
        """Ensure the proxy is running, start it if needed"""
        if self.is_proxy_running():
            return True
        
        return self.start_proxy()
    
    def stop_proxy(self):
        """Stop the Argo proxy if it was started by this manager"""
        with self._lock:
            if self.proxy_process and self.proxy_process.poll() is None:
                try:
                    # Send SIGTERM to the process group
                    if hasattr(os, 'killpg'):
                        os.killpg(os.getpgid(self.proxy_process.pid), signal.SIGTERM)
                    else:
                        self.proxy_process.terminate()
                    
                    # Wait a bit for graceful shutdown
                    try:
                        self.proxy_process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't respond
                        if hasattr(os, 'killpg'):
                            os.killpg(os.getpgid(self.proxy_process.pid), signal.SIGKILL)
                        else:
                            self.proxy_process.kill()
                    
                    print("✓ Argo proxy stopped")
                except Exception as e:
                    print(f"Warning: Error stopping proxy: {e}")
                finally:
                    self.proxy_process = None
    
    def is_argo_config(self, config_file: str) -> bool:
        """Check if a configuration file contains Argo server configurations"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            if not config or 'servers' not in config:
                return False
            
            # Check if any server has a localhost Argo configuration
            for server in config['servers']:
                if isinstance(server, dict):
                    api_base = server.get('openai_api_base', '')
                    if 'localhost' in api_base and 'argo' in server.get('server', '').lower():
                        return True
                    # Also check if the model name starts with 'argo:'
                    model_name = server.get('openai_model', '')
                    if model_name.startswith('argo:'):
                        return True
            
            return False
        except Exception:
            return False
    
    def check_and_start_proxy_if_needed(self, config_file: str) -> bool:
        """
        Check if the configuration uses Argo and start the proxy if needed.
        
        Returns:
            bool: True if Argo proxy is ready, False if there was an error
        """
        if not self.is_argo_config(config_file):
            return True  # Not an Argo config, nothing to do
        
        print("Argo configuration detected - checking proxy status...")
        
        # Check current status
        status = self.get_status(config_file)
        
        if status.is_running:
            print("✓ Argo proxy is already running")
            return True
        
        if not status.executable_found:
            print("✗ Argo proxy executable not found")
            print("Please install argo-proxy or ensure it's in your PATH")
            return False
        
        # Try to start the proxy
        print("Starting Argo proxy...")
        return self.ensure_proxy_running()
    
    def get_status(self, config_file: Optional[str] = None) -> ArgoStatus:
        """Get the current status of the Argo proxy and configuration"""
        is_running = self.is_proxy_running()
        executable_found = self.find_argo_proxy_executable() is not None
        is_argo_config = self.is_argo_config(config_file) if config_file else False
        
        return ArgoStatus(
            is_running=is_running,
            host=self.host,
            port=self.port,
            executable_found=executable_found,
            managed_by_us=self.proxy_process is not None and self.proxy_process.poll() is None,
            config_file=config_file,
            is_argo_config=is_argo_config
        )
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.stop_proxy()


# Global Argo manager instance
_argo_manager: Optional[ArgoManager] = None


def get_argo_manager() -> ArgoManager:
    """Get the global Argo manager instance"""
    global _argo_manager
    if _argo_manager is None:
        _argo_manager = ArgoManager()
    return _argo_manager


def ensure_argo_proxy_running() -> bool:
    """Ensure the Argo proxy is running, start it if needed"""
    manager = get_argo_manager()
    return manager.ensure_proxy_running()


def check_argo_proxy_status() -> ArgoStatus:
    """Check the current status of the Argo proxy"""
    manager = get_argo_manager()
    return manager.get_status()


def check_and_start_argo_proxy_if_needed(config_file: str) -> bool:
    """Check if config uses Argo and start proxy if needed"""
    manager = get_argo_manager()
    return manager.check_and_start_proxy_if_needed(config_file)


def cleanup_argo_proxy():
    """Clean up the Argo proxy if it was started by this manager"""
    global _argo_manager
    if _argo_manager:
        _argo_manager.stop_proxy()
        _argo_manager = None


# Register cleanup on exit
import atexit
atexit.register(cleanup_argo_proxy)


# Wrapper functions for automatic Argo proxy management
def create_openai_client(**kwargs):
    """Create an OpenAI client with automatic Argo proxy management"""
    base_url = kwargs.get('base_url')
    
    # Handle CELS bridge service SSL issues
    if base_url and 'argo-bridge.cels.anl.gov' in base_url:
        import httpx
        # Create client with SSL verification disabled for CELS bridge
        kwargs['http_client'] = httpx.Client(verify=False)
        print("Note: Using CELS Argo bridge with SSL verification disabled")
    elif base_url and 'localhost' in base_url:
        # This is a local Argo configuration, try to start proxy
        manager = get_argo_manager()
        if not manager.ensure_proxy_running():
            print("Warning: Failed to start Argo proxy. API calls may fail.")
    
    from openai import OpenAI
    return OpenAI(**kwargs)

def create_async_openai_client(**kwargs):
    """Create an AsyncOpenAI client with automatic Argo proxy management"""
    base_url = kwargs.get('base_url')
    
    # Handle CELS bridge service SSL issues
    if base_url and 'argo-bridge.cels.anl.gov' in base_url:
        import httpx
        # Create client with SSL verification disabled for CELS bridge
        kwargs['http_client'] = httpx.AsyncClient(verify=False)
        print("Note: Using CELS Argo bridge with SSL verification disabled")
    elif base_url and 'localhost' in base_url:
        # This is a local Argo configuration, try to start proxy
        manager = get_argo_manager()
        if not manager.ensure_proxy_running():
            print("Warning: Failed to start Argo proxy. API calls may fail.")
    
    from openai import AsyncOpenAI
    return AsyncOpenAI(**kwargs)
