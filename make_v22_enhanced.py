#!/usr/bin/env python3
# make_v22_enhanced.py - Enhanced Question Generation Script with High-Performance Parallel Processing
# 
# Enhanced version of make_v22.py with:
# - Dramatically increased parallelism (50-200+ concurrent API calls)
# - Robust failure tracking and retry mechanisms
# - Enhanced checkpoint/resume functionality  
# - Connection pooling optimization
# - Real-time performance monitoring

import os
import sys
import json
import re
import time
import random
import asyncio
import uuid
import httpx
from openai import OpenAI, AsyncOpenAI
import PyPDF2
import spacy
import yaml
import argparse
import pathlib
import signal
import hashlib
import curses
import threading
import queue
import atexit
import multiprocessing
import zipfile
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum
from tqdm import tqdm
from dataclasses import dataclass, asdict
from collections import defaultdict
import backoff

# Import Argo utilities to enable automatic proxy management
from argo_utils import create_openai_client, create_async_openai_client


def create_question_output(chunk_id: str, question_data: dict, chunk_text: str = None, 
                          source_info: dict = None, processing_time: float = None) -> dict:
    """
    Standardized output format for all question generation systems
    
    Args:
        chunk_id: Unique chunk identifier
        question_data: Dict with 'question', 'answer', 'type', etc.
        chunk_text: Original text chunk content
        source_info: Dict with source file information (path, size, type, etc.)
        processing_time: Time taken to process this chunk
    
    Returns:
        Standardized question output dictionary matching original format
    """
    output = {
        "question": question_data.get("question", ""),
        "answer": question_data.get("answer", ""),
        "text": chunk_text or "",
        "type": question_data.get("type", "unknown"),
    }
    
    # Add source file information if provided
    if source_info:
        output.update({
            "source_file_path": source_info.get("source_file_path", ""),
            "source_relative_path": source_info.get("source_relative_path", ""),
            "source_filename": source_info.get("source_filename", ""),
            "source_file_size": source_info.get("source_file_size", 0),
            "source_file_type": source_info.get("source_file_type", ""),
            "source_last_modified": source_info.get("source_last_modified", 0.0),
        })
    
    # Add any additional question-specific fields
    for key, value in question_data.items():
        if key not in ["question", "answer", "type"]:
            output[key] = value
            
    return output


def load_model_config(model_name: str, config_file: str = "argo_model_servers.yaml") -> tuple:
    """
    Load model configuration from YAML file
    Returns: (api_key, base_url, model_name)
    """
    import os
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Find the model configuration
    for server in config.get("servers", []):
        if server.get("shortname") == model_name:
            api_key = server.get("openai_api_key", "")
            base_url = server.get("openai_api_base", "")
            actual_model = server.get("openai_model", model_name)
            
            # Handle environment variables in API key
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                api_key = os.environ.get(env_var, api_key)
            
            return api_key, base_url, actual_model
    
    raise ValueError(f"Model '{model_name}' not found in {config_file}")


@dataclass
class FailureRecord:
    """Record of a failed API call or chunk processing"""
    chunk_id: str
    error_type: str
    error_message: str
    timestamp: float
    retry_count: int = 0
    step_name: str = ""
    api_call_details: Dict = None


@dataclass 
class ProcessingStats:
    """Real-time processing statistics"""
    total_chunks: int = 0
    completed_chunks: int = 0
    failed_chunks: int = 0
    retried_chunks: int = 0
    api_calls_made: int = 0
    api_calls_failed: int = 0
    api_calls_retried: int = 0
    start_time: float = 0
    last_update_time: float = 0
    
    def get_completion_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.completed_chunks / self.total_chunks
    
    def get_failure_rate(self) -> float:
        if self.api_calls_made == 0:
            return 0.0
        return self.api_calls_failed / self.api_calls_made
    
    def get_elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def get_estimated_time_remaining(self) -> float:
        completion_rate = self.get_completion_rate()
        if completion_rate <= 0:
            return float('inf')
        elapsed = self.get_elapsed_time()
        return elapsed * (1 - completion_rate) / completion_rate


class EnhancedFailureTracker:
    """Advanced failure tracking and retry management"""
    
    def __init__(self, max_retries: int = 3, failure_log_file: str = None):
        self.max_retries = max_retries
        self.failures: Dict[str, FailureRecord] = {}
        self.failure_log_file = failure_log_file or f"failures_{int(time.time())}.json"
        self.lock = threading.Lock()
        
    def record_failure(self, chunk_id: str, error_type: str, error_message: str, 
                      step_name: str = "", api_call_details: Dict = None):
        """Record a failure for later retry"""
        with self.lock:
            if chunk_id not in self.failures:
                self.failures[chunk_id] = FailureRecord(
                    chunk_id=chunk_id,
                    error_type=error_type,
                    error_message=error_message,
                    timestamp=time.time(),
                    step_name=step_name,
                    api_call_details=api_call_details or {}
                )
            else:
                # Update existing failure record
                self.failures[chunk_id].retry_count += 1
                self.failures[chunk_id].timestamp = time.time()
                self.failures[chunk_id].error_message = error_message
    
    def should_retry(self, chunk_id: str) -> bool:
        """Check if a chunk should be retried"""
        with self.lock:
            if chunk_id not in self.failures:
                return False
            return self.failures[chunk_id].retry_count < self.max_retries
    
    def get_failed_chunks(self) -> List[str]:
        """Get list of chunks that have failed but can still be retried"""
        with self.lock:
            return [chunk_id for chunk_id, record in self.failures.items() 
                   if record.retry_count < self.max_retries]
    
    def get_permanently_failed_chunks(self) -> List[str]:
        """Get list of chunks that have exceeded max retries"""
        with self.lock:
            return [chunk_id for chunk_id, record in self.failures.items() 
                   if record.retry_count >= self.max_retries]
    
    def save_failure_log(self):
        """Save failure log to disk"""
        with self.lock:
            failure_data = {
                'timestamp': time.time(),
                'total_failures': len(self.failures),
                'retryable_failures': len(self.get_failed_chunks()),
                'permanent_failures': len(self.get_permanently_failed_chunks()),
                'failures': {chunk_id: asdict(record) for chunk_id, record in self.failures.items()}
            }
            
            with open(self.failure_log_file, 'w') as f:
                json.dump(failure_data, f, indent=2)
    
    def load_failure_log(self, log_file: str):
        """Load failure log from disk"""
        with open(log_file, 'r') as f:
            failure_data = json.load(f)
        
        with self.lock:
            for chunk_id, record_dict in failure_data.get('failures', {}).items():
                self.failures[chunk_id] = FailureRecord(**record_dict)


class EnhancedAPIManager:
    """High-performance API manager with connection pooling and retry logic"""
    
    def __init__(self, model_name: str, max_concurrent_calls: int = 100, 
                 timeout: int = 60, max_retries: int = 2,
                 api_key: str = None, base_url: str = None):
        self.model_name = model_name
        self.max_concurrent_calls = max_concurrent_calls
        self.timeout = timeout
        self.max_retries = max_retries
        self.api_key = api_key
        self.base_url = base_url
        
        # Semaphore to limit concurrent API calls
        self.semaphore = asyncio.Semaphore(max_concurrent_calls)
        
        self.async_client = None
        self.stats = ProcessingStats()
        # In-flight API tracking
        self.in_flight_api = 0
        self._in_flight_lock = asyncio.Lock()
        self._in_flight_tokens = {}  # token -> start_ts
        # Global backoff and status buckets
        self.global_backoff_until = 0.0
        self._status_window_started = time.time()
        self._status_buckets = {
            '2xx': 0,
            '429': 0,
            '4xx': 0,
            '5xx': 0,
            'other': 0,
        }

    async def _increment_in_flight(self, token: str, start_ts: float) -> None:
        async with self._in_flight_lock:
            self.in_flight_api += 1
            self._in_flight_tokens[token] = start_ts

    async def _decrement_in_flight(self, token: str) -> None:
        async with self._in_flight_lock:
            if token in self._in_flight_tokens:
                del self._in_flight_tokens[token]
            if self.in_flight_api > 0:
                self.in_flight_api -= 1

    def get_in_flight_api(self) -> int:
        return self.in_flight_api

    def get_longest_wait_seconds(self) -> float:
        if not self._in_flight_tokens:
            return 0.0
        oldest = min(self._in_flight_tokens.values())
        return max(0.0, time.time() - oldest)
    
    def _status_window_maybe_reset(self) -> None:
        if (time.time() - self._status_window_started) >= 30:
            self._status_buckets = {k: 0 for k in self._status_buckets}
            self._status_window_started = time.time()
    
    def get_status_window_buckets(self) -> dict:
        return dict(self._status_buckets)
    
    def _bucket_from_exception(self, exc: Exception) -> str:
        msg = str(exc).lower()
        if '429' in msg or 'rate limit' in msg or 'too many requests' in msg:
            return '429'
        if 'status 5' in msg or ' 5' in msg and 'status' in msg:
            return '5xx'
        if 'status 4' in msg or ' 4' in msg and 'status' in msg:
            return '4xx'
        return 'other'
        
    async def __aenter__(self):
        """Async context manager entry"""
        # Build client config - let create_async_openai_client handle http_client for CELS bridge
        client_config = {
            "timeout": self.timeout
        }
        
        # Add API key and base URL if provided
        if self.api_key:
            client_config["api_key"] = self.api_key
        if self.base_url:
            client_config["base_url"] = self.base_url
            
        self.async_client = create_async_openai_client(**client_config)
        self.stats.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.async_client:
            await self.async_client.close()

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=5,  # More retries
        max_time=30,  # Shorter total time
        base=2,       # Faster exponential growth
        jitter=backoff.random_jitter,
        giveup=lambda e: (
            "authentication" in str(e).lower()
            # Removed rate limit from giveup - we WANT to retry 429s!
        ),
        on_backoff=lambda details: None,
        on_giveup=lambda details: None
    )
    async def make_api_call(self, messages: List[Dict], **kwargs) -> Dict:
        """Make a single API call with retry logic and rate limiting"""
        async with self.semaphore:  # Limit concurrent calls
            self.stats.api_calls_made += 1
            token = uuid.uuid4().hex
            start_ts = time.time()
            await self._increment_in_flight(token, start_ts)
            
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **kwargs
            )
            
            # Convert to dict for compatibility
            if hasattr(response, 'model_dump'):
                result = response.model_dump()
            else:
                result = response
            # Success bucket
            self._status_buckets['2xx'] += 1
            self._status_window_maybe_reset()
            return result
    
    async def process_chunk_parallel(self, chunk_id: str, chunk_text: str, 
                                   question_type: str, num_answers: int = 7, 
                                   min_score: int = 7) -> Dict:
        """Process a single chunk with parallel API calls where possible"""
        start_time = time.time()
        
        if 0: print(f"DEBUG: process_chunk_parallel: Starting for chunk {chunk_id}, type={question_type}")
        # Generate question based on type - ALWAYS use original make_v22.py functions
        # These functions already handle content relevance, quality evaluation, etc.
        if question_type in ["multiple_choice", "mc"]:
            if 0: print(f"DEBUG: process_chunk_parallel: Calling generate_multiple_choice_question_async for {chunk_id}")
            result = await self.generate_multiple_choice_question_async(
                chunk_id, chunk_text, num_answers, min_score
            )
        elif question_type in ["free_form", "ff", "qa"]:
            result = await self.generate_free_form_question_async(
                chunk_id, chunk_text, min_score
            )
        elif question_type in ["reasoning_trace", "rt"]:
            result = await self.generate_reasoning_trace_question_async(
                chunk_id, chunk_text, min_score
            )
        else:
            # Default to multiple choice for unknown types
            result = await self.generate_multiple_choice_question_async(
                chunk_id, chunk_text, num_answers, min_score
            )
        
        # The original functions return results with all necessary fields
        # Add processing time, ensure chunk text is included
        if result:
            result['processing_time'] = time.time() - start_time
            if 'text' not in result:
                result['text'] = chunk_text
        
        return result

    async def generate_multiple_choice_question_async(self, chunk_id: str, chunk_text: str, 
                                                    num_answers: int, min_score: int) -> Dict:
        """Async wrapper for generate_multiple_choice_qa_pairs from make_v22.py"""
        if 0: print(f"DEBUG: generate_multiple_choice_question_async: Starting for {chunk_id}")
        if 0: print(f"DEBUG: Input parameters - chunk_id: {chunk_id}, text_length: {len(chunk_text)}, model: {self.model_name}, num_answers: {num_answers}, min_score: {min_score}")

        # Import the function from make_v22.py
        if 0: print(f"DEBUG: generate_multiple_choice_question_async: About to import for {chunk_id}")
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from make_v22 import generate_multiple_choice_qa_pairs
        import make_v22
        if 0: print(f"DEBUG: generate_multiple_choice_question_async: Import successful for {chunk_id}")

        # Set up required globals that the original function expects
        if 0: print(f"DEBUG: Setting up globals for {chunk_id}")
        if not hasattr(make_v22, '_exit_requested'):
            make_v22._exit_requested = False
            if 0: print(f"DEBUG: Set _exit_requested = False for {chunk_id}")
            
        # Initialize the global _openai_client that the original functions expect
        if 0: print(f"DEBUG: Checking _openai_client for {chunk_id}")
        if not hasattr(make_v22, '_openai_client') or make_v22._openai_client is None:
            if 0: print(f"DEBUG: _openai_client not found, creating new one for {chunk_id}")
            # Create a client using the same configuration as the enhanced system
            from argo_utils import create_openai_client
            client_config = {
                'api_key': self.api_key,
                'base_url': self.base_url
            }
            if 0: print(f"DEBUG: Client config for {chunk_id}: {client_config}")
            make_v22._openai_client = create_openai_client(**client_config)
            if 0: print(f"DEBUG: Created _openai_client for {chunk_id}: {type(make_v22._openai_client)}")
        else:
            if 0: print(f"DEBUG: _openai_client already exists for {chunk_id}: {type(make_v22._openai_client)}")

        # Call the original function (it's synchronous)
        if 0: print(f"DEBUG: About to call generate_multiple_choice_qa_pairs for {chunk_id}")
        loop = asyncio.get_event_loop()
        if 0: print(f"DEBUG: Got event loop for {chunk_id}: {type(loop)}")
        result = await loop.run_in_executor(
            None,
            generate_multiple_choice_qa_pairs,
            chunk_id, chunk_text, self.model_name, num_answers, min_score
        )
        if 0: print(f"DEBUG: generate_multiple_choice_qa_pairs returned for {chunk_id}: {type(result)}, status={result.get('status') if result else None}")
        
        # Add detailed logging for ALL return values to see what's happening
        if 0: print(f"DEBUG: FULL RESULT for {chunk_id}: {result}")
        
        # Add detailed logging for error cases
        if result and result.get('status') == 'error':
            if 0: print(f"DEBUG: ERROR DETAILS for {chunk_id}: {result}")
            if 'error' in result:
                if 0: print(f"DEBUG: Error message: {result['error']}")
        
        return result
    
    async def generate_free_form_question_async(self, chunk_id: str, chunk_text: str, 
                                                min_score: int) -> Dict:
        """Async wrapper for generate_free_form_qa_pairs from make_v22.py"""
        # Import the function from make_v22.py
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from make_v22 import generate_free_form_qa_pairs
        import make_v22
        
        # Set up required globals that the original function expects
        if not hasattr(make_v22, '_exit_requested'):
            make_v22._exit_requested = False
            
        # Initialize the global _openai_client that the original functions expect
        if not hasattr(make_v22, '_openai_client') or make_v22._openai_client is None:
            # Create a client using the same configuration as the enhanced system
            from argo_utils import create_openai_client
            client_config = {
                'api_key': self.api_key,
                'base_url': self.base_url
            }
            make_v22._openai_client = create_openai_client(**client_config)
        
        # Call the original function (it's synchronous)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            generate_free_form_qa_pairs, 
            chunk_id, chunk_text, self.model_name, min_score
        )
        
        # Add detailed logging for ALL return values to see what's happening
        if 0: print(f"DEBUG: FULL RESULT for {chunk_id}: {result}")
        
        # Add detailed logging for error cases
        if result and result.get('status') == 'error':
            if 0: print(f"DEBUG: ERROR DETAILS for {chunk_id}: {result}")
            if 'error' in result:
                if 0: print(f"DEBUG: Error message: {result['error']}")
        
        # Return the original result - source info will be added later in process_single_chunk
        return result
    
    async def generate_reasoning_trace_question_async(self, chunk_id: str, chunk_text: str, 
                                                    min_score: int) -> Dict:
        """Async wrapper for generate_reasoning_trace_pairs from make_v22.py"""
        # Import the function from make_v22.py
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from make_v22 import generate_reasoning_trace_pairs
        import make_v22
        
        # Set up required globals that the original function expects
        if not hasattr(make_v22, '_exit_requested'):
            make_v22._exit_requested = False
            
        # Initialize the global _openai_client that the original functions expect
        if not hasattr(make_v22, '_openai_client') or make_v22._openai_client is None:
            # Create a client using the same configuration as the enhanced system
            from argo_utils import create_openai_client
            client_config = {
                'api_key': self.api_key,
                'base_url': self.base_url
            }
            make_v22._openai_client = create_openai_client(**client_config)
        
        # Call the original function (it's synchronous)
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            generate_reasoning_trace_pairs, 
            chunk_id, chunk_text, self.model_name, min_score
        )
        
        # Add detailed logging for error cases
        if result and result.get('status') == 'error':
            if 0: print(f"DEBUG: ERROR DETAILS for {chunk_id}: {result}")
            if 'error' in result:
                if 0: print(f"DEBUG: Error message: {result['error']}")
        
        # Return the original result - source info will be added later in process_single_chunk
        return result

    # End of EnhancedAPIManager class


class EnhancedChunkProcessor:
    """High-performance chunk processor with failure tracking and retry logic"""
    
    def __init__(self, model_name: str, max_concurrent_calls: int = 100, 
                 max_retries: int = 3, failure_log_file: str = None):
        self.model_name = model_name
        self.max_concurrent_calls = max_concurrent_calls
        self.failure_tracker = EnhancedFailureTracker(max_retries, failure_log_file)
        self.stats = ProcessingStats()
        self.output_file = None  # Will be set during processing
        self.checkpoint_lock = asyncio.Lock()  # Lock for checkpoint file access
        self._file_map = {}  # Will store chunk_id -> source_info mapping
        
    def _get_source_info_from_chunk_id(self, chunk_id: str) -> dict:
        """Get source file information from chunk_id"""
        # Extract file_id from chunk_id (format: file_id_chunk_number)
        file_id = chunk_id.split('_')[0] if '_' in chunk_id else chunk_id
        
        if file_id in self._file_map:
            file_info = self._file_map[file_id]
            return {
                "source_file_path": file_info.get('file_path', ''),
                "source_relative_path": file_info.get('relative_path', ''),
                "source_filename": file_info.get('filename', ''),
                "source_file_size": file_info.get('size', 0),
                "source_file_type": file_info.get('type', ''),
                "source_last_modified": file_info.get('last_modified', 0.0),
            }
        else:
            return {
                "source_file_path": "unknown",
                "source_relative_path": "unknown", 
                "source_filename": "unknown",
                "source_file_size": 0,
                "source_file_type": "unknown",
                "source_last_modified": 0.0
            }
        
    async def save_results_incrementally(self, new_results: List[Dict]):
        """Save new results to output file incrementally with async lock"""
        async with self.checkpoint_lock:  # Use same lock for all file I/O
            # Load existing questions if file exists
            existing_questions = []
            if os.path.exists(self.output_file):
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_questions = json.load(f)
                    if not isinstance(existing_questions, list):
                        existing_questions = []
            
            # Add new results
            existing_questions.extend(new_results)
            
            # Write back to file
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_questions, f, ensure_ascii=False, indent=2)
        
    async def process_chunks_enhanced(self, chunk_ids: List[str], chunks_dir: str,
                                    question_type: str, num_answers: int = 7, 
                                    min_score: int = 7, checkpoint_manager=None, output_file: str = None) -> List[Dict]:
        """Process chunks with enhanced parallelism and failure tracking"""
        
        # Load any existing failures first (auto-retry)
        if os.path.exists(self.failure_tracker.failure_log_file):
            self.failure_tracker.load_failure_log(self.failure_tracker.failure_log_file)
            print(f"📁 Found existing failure log: {self.failure_tracker.failure_log_file}")
        
        # Get failed chunks that can be retried (automatic retry)
        retryable_chunks = self.failure_tracker.get_failed_chunks()
        if retryable_chunks:
            print(f"🔄 Auto-retrying {len(retryable_chunks)} failed chunks from previous runs")
        
        # Filter out already processed chunks
        if checkpoint_manager:
            chunk_ids = [c_id for c_id in chunk_ids if not checkpoint_manager.is_chunk_processed(c_id)]
        
        # Combine new chunks with retryable failed chunks (automatic retry)
        all_chunks = list(set(chunk_ids + retryable_chunks))
        
        self.stats.total_chunks = len(all_chunks)
        self.stats.start_time = time.time()
        self.output_file = output_file  # Store for incremental saving
        
        print(f"Processing {len(all_chunks)} chunks with up to {self.max_concurrent_calls} concurrent API calls")
        print(f"Including {len(retryable_chunks)} chunks from previous failures")
        # Load model configuration from YAML
        api_key, base_url, actual_model_name = load_model_config(self.model_name)
        
        results = []
        
        async with EnhancedAPIManager(
            actual_model_name, 
            self.max_concurrent_calls,
            api_key=api_key,
            base_url=base_url
        ) as api_manager:
            # Create semaphore for chunk-level concurrency (use max_concurrent_calls)
            chunk_semaphore = asyncio.Semaphore(min(self.max_concurrent_calls, len(all_chunks)))
            # Watchdog and health tracking
            last_completion_ts = time.time()
            last_health_write_ts = time.time()
            last_stats_print_ts = time.time()
            health_file = "enhanced_health.json"
            
            async def process_single_chunk(chunk_id: str) -> Optional[Dict]:
                """Process a single chunk with error handling"""
                if 0: print(f"DEBUG: ENTERING process_single_chunk for {chunk_id}")
                async with chunk_semaphore:
                    if 0: print(f"DEBUG: ACQUIRED semaphore for {chunk_id}")
                    if 0: print(f"DEBUG: Processing chunk {chunk_id}")
                    # Read chunk file
                    chunk_subdir = chunk_id[:2]
                    chunk_file_path = os.path.join(chunks_dir, chunk_subdir, f"{chunk_id}.txt")
                    
                    with open(chunk_file_path, 'r', encoding='utf-8') as f:
                        chunk_text = f.read()
                    if 0: print(f"DEBUG: Read chunk {chunk_id}, length: {len(chunk_text)}")
                    
                    if 0: print(f"DEBUG: About to call process_chunk_parallel for {chunk_id}")
                    result = await api_manager.process_chunk_parallel(
                        chunk_id, chunk_text, question_type, num_answers, min_score
                    )
                    if 0: print(f"DEBUG: process_chunk_parallel returned for {chunk_id}: {type(result)}, status={result.get('status') if result else None}")
                    
                    # Add source file information from the file map
                    if result and hasattr(self, '_file_map') and self._file_map:
                        source_info = self._get_source_info_from_chunk_id(chunk_id)
                        if source_info:
                            result.update(source_info)
                    
                    if result.get('status') == 'error':
                        self.failure_tracker.record_failure(
                            chunk_id, "processing", result.get('error', 'Unknown error'), 
                            "chunk_processing"
                        )
                        return None
                    elif result.get('status') in ['success', 'completed']:
                        self.stats.completed_chunks += 1
                        
                        # Save to checkpoint with async lock to prevent blocking
                        if checkpoint_manager:
                            async with self.checkpoint_lock:
                                checkpoint_manager.update_processed_chunk(chunk_id, result)
                        
                        return result
                    else:
                        # Filtered or low quality - not an error but no result
                        self.stats.completed_chunks += 1
                        
                        # Save filtered chunks to checkpoint so they aren't reprocessed
                        if checkpoint_manager and result:
                            async with self.checkpoint_lock:
                                checkpoint_manager.update_processed_chunk(chunk_id, result)
                        
                        return None
            
            # Create progress bar
            with tqdm(total=len(all_chunks), desc="Processing chunks") as pbar:

                # Process chunks concurrently using a rolling window with auto-scaling
                window = min(self.max_concurrent_calls, len(all_chunks))
                in_flight: List[asyncio.Task] = []
                in_flight_start_times: Dict[asyncio.Task, float] = {}  # Track when each task started
                chunk_iter = iter(all_chunks)
                last_adjust_ts = time.time()
                last_status_print = time.time()

                # Prime the window
                # Prime the window with initial tasks
                if 0: print(f"DEBUG: About to prime window with up to {min(window, len(all_chunks))} tasks from {len(all_chunks)} total chunks")
                for i in range(min(window, len(all_chunks))):
                    cid = next(chunk_iter)
                    if 0: print(f"DEBUG: Got chunk ID from iterator: {cid}")
                    if 0: print(f"DEBUG: Creating task for chunk {cid}")
                    task = asyncio.create_task(process_single_chunk(cid))
                    if 0: print(f"DEBUG: Created task object: {task}")
                    in_flight.append(task)
                    in_flight_start_times[task] = time.time()  # Track start time
                # Process tasks with minimal wait time for immediate response
                while in_flight:
                    if 0: print(f"DEBUG: About to wait on {len(in_flight)} tasks")
                    done, pending = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED, timeout=1.0)
                    if 0: print(f"DEBUG: asyncio.wait returned {len(done)} done, {len(pending)} pending")
                    in_flight = list(pending)
                    
                    # Process all completed tasks immediately
                    new_results = []
                    for task in done:
                        # Clean up start time tracking
                        if task in in_flight_start_times:
                            del in_flight_start_times[task]
                        
                        res = await task
                        if res:
                            results.append(res)
                            new_results.append(res)
                        pbar.update(1)
                    
                    # Save new results to output file immediately
                    if new_results and self.output_file:
                        await self.save_results_incrementally(new_results)
                    
                    # Refill window immediately
                    while len(in_flight) < window:
                        cid = next(chunk_iter)
                        if 0: print(f"DEBUG: Creating refill task for chunk {cid}")
                        task = asyncio.create_task(process_single_chunk(cid))
                        in_flight.append(task)
                        in_flight_start_times[task] = time.time()  # Track start time
                    
                    if done:
                        pass  # Stats could be logged here if needed
                    
                    # Print status every 30 seconds to show stuck tasks
                    current_time = time.time()
                    if current_time - last_status_print >= 30:
                        self._print_in_flight_status(in_flight, in_flight_start_times, current_time)
                        last_status_print = current_time
        
        # Final statistics
        self.stats.end_time = time.time()
        print(f"Processed {len(results)} chunks successfully")
        print(f"Failed chunks: {len(self.failure_tracker.get_failed_chunks())}")
        print(f"Permanently failed: {len(self.failure_tracker.get_permanently_failed_chunks())}")
        print(f"Total processing time: {self.stats.get_elapsed_time():.1f}s")
        print(f"Failure log saved to: {self.failure_tracker.failure_log_file}")
        
        return results
    
    def _print_in_flight_status(self, in_flight: List[asyncio.Task], start_times: Dict[asyncio.Task, float], current_time: float):
        """Print status of in-flight tasks to show stuck ones"""
        if not in_flight:
            return
            
        print(f"\n🔄 IN-FLIGHT STATUS: {len(in_flight)} tasks running")
        print("=" * 60)
        
        # Group tasks by duration
        duration_groups = {
            "0-1 min": [],
            "1-5 min": [],
            "5-15 min": [],
            "15+ min": []
        }
        
        for task in in_flight:
            if task in start_times:
                duration = current_time - start_times[task]
                if duration < 60:
                    duration_groups["0-1 min"].append(task)
                elif duration < 300:
                    duration_groups["1-5 min"].append(task)
                elif duration < 900:
                    duration_groups["5-15 min"].append(task)
                else:
                    duration_groups["15+ min"].append(task)
        
        # Print summary
        for group_name, tasks in duration_groups.items():
            if tasks:
                print(f"  {group_name}: {len(tasks)} tasks")
        
        # Show longest-running tasks
        if in_flight:
            longest_running = max(in_flight, key=lambda t: current_time - start_times.get(t, 0))
            if longest_running in start_times:
                max_duration = current_time - start_times[longest_running]
                print(f"  Longest running: {max_duration:.1f}s")
        
        print("=" * 60)
    
# Enhanced main processing function
async def process_chunks_with_enhanced_parallel_workers(
    chunk_ids: List[str], chunks_dir: str, model_name: str,
    question_type: str, num_answers: int, min_score: int,
    checkpoint_manager, output_file: str, max_concurrent_calls: int = 100,
    file_map: dict = None
) -> List[Dict]:
    """
    Enhanced chunk processing with high parallelism and failure tracking
    
    Args:
        chunk_ids: List of chunk IDs to process
        chunks_dir: Directory containing chunk files
        model_name: Model name for API calls
        question_type: Type of questions to generate
        num_answers: Number of answers for multiple choice
        min_score: Minimum quality score
        checkpoint_manager: Checkpoint manager for progress saving
        output_file: Output file path
        max_concurrent_calls: Maximum concurrent API calls
    """
    
    processor = EnhancedChunkProcessor(
        model_name=model_name,
        max_concurrent_calls=max_concurrent_calls,
        failure_log_file=f"failures_{os.path.basename(output_file)}.json"
    )
    
    # Set the file map if provided
    if file_map:
        processor._file_map = file_map
    
    results = await processor.process_chunks_enhanced(
        chunk_ids=chunk_ids,
        chunks_dir=chunks_dir, 
        question_type=question_type,
        num_answers=num_answers,
        min_score=min_score,
        checkpoint_manager=checkpoint_manager,
        output_file=output_file
    )
    
    return results


def run_enhanced_processing(chunk_ids: List[str], chunks_dir: str, model_name: str,
                        question_type: str, num_answers: int, min_score: int,
                        checkpoint_manager, output_file: str, max_concurrent_calls: int = 100,
                        file_map: dict = None):
    """
    Synchronous wrapper for enhanced async processing
    """
    return asyncio.run(
        process_chunks_with_enhanced_parallel_workers(
            chunk_ids, chunks_dir, model_name, question_type, 
            num_answers, min_score, checkpoint_manager, output_file, max_concurrent_calls, file_map
        )
    )


if __name__ == "__main__":
    print("Enhanced Question Generation System")
    print("This module provides high-performance parallel processing capabilities.")
    print("Import and use run_enhanced_processing() function to replace the original processing.")
