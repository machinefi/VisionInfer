import subprocess
import os
import signal
import psutil
import time
from ..constants import ollama_process
from ..utils import logger

def start_ollama_serve():
    global ollama_process
    try:
        if is_ollama_running():
            print("✅ Ollama service is already running")
            return True
        
        print("🚀 Starting Ollama service...")
        ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            shell=False
        )
        time.sleep(2)
        
        if ollama_process.poll() is None and is_ollama_running():
            print("✅ Ollama service started successfully")
            return True
        else:
            print("❌ Failed to start Ollama service")
            return False
    except Exception as e:
        print(f"❌ Exception starting Ollama: {e}")
        return False

def is_ollama_running():
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if proc.info['name'] == 'ollama' and 'serve' in proc.info['cmdline']:
                return True
        return False
    except:
        return False

def stop_ollama_serve():
    global ollama_process
    current_uid = os.getuid() 
    
    if ollama_process and ollama_process.poll() is None:
        try:
            ollama_process.terminate()
            ollama_process.wait(timeout=10)
            logger.info(f"Stopped self-started Ollama process (PID: {ollama_process.pid})")
        except Exception as e:
            logger.warning(f"Failed to stop self-started Ollama process: {e}")
        finally:
            ollama_process = None
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'uids']):
        try:
            proc_uids = proc.info['uids']
            if not proc_uids:
                continue

            if proc_uids.real != current_uid:
                continue
            
            is_ollama = False
            if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                is_ollama = True
            if proc.info['cmdline'] and 'ollama serve' in ' '.join(proc.info['cmdline']).lower():
                is_ollama = True
            
            if is_ollama:
                pid = proc.info['pid']
                if pid == os.getpid() or pid == 1: 
                    continue
                proc.terminate()
                proc.wait(timeout=5)
                logger.info(f"Cleaned residual Ollama process (PID: {pid}, user-owned)")
        except psutil.NoSuchProcess:
            continue 
        except psutil.AccessDenied:
            continue
        except Exception as e:
            logger.warning(f"Failed to clean user-owned Ollama process: {e}")

def get_ollama_usage_data():
    return {"status": "Not implemented", "models": []}

def get_ollama_inference_perf(model_name):
    return {"status": "Not implemented", "model": model_name}

def print_ollama_usage():
    """Print Ollama API usage data (compatible with low versions)"""
    print("\n" + "="*60)
    print("📊 Ollama API Usage Monitoring (compatible with low versions)")
    print("="*60)
    usage_data = get_ollama_usage_data()
    
    for key, value in usage_data.items():
        print(f"\n🔹 {key}:")
        if isinstance(value, list):
            if len(value) == 0:
                print("  No data")
            else:
                for idx, item in enumerate(value):
                    print(f"  [{idx+1}] Model name: {item.get('name', 'Unknown')}")
                    if "pid" in item:
                        print(f"     PID: {item.get('pid', 'Not obtained')}")
                    if "size" in item and item["size"] > 0:
                        print(f"     Size: {round(item.get('size', 0)/1024/1024/1024, 2)}GB")
                    if "modified_at" in item and item["modified_at"] != "Unknown":
                        print(f"     Last updated: {item.get('modified_at', 'Unknown')}")
                    if "ports" in item and item["ports"] != "None":
                        print(f"     Ports used: {item.get('ports', 'None')}")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
    print("\n" + "="*60)

def print_ollama_perf(model_name):
    """Print inference performance data (standard fields from official docs)"""
    print("\n" + "="*80)
    print("📊 Ollama /api/generate Inference Performance Monitoring (official standard fields)")
    print("="*80)
    perf_data = get_ollama_inference_perf(model_name)
    
    for key, value in perf_data.items():
        print(f"\n🔹 {key}:")
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"  {value}")
    print("\n" + "="*80)