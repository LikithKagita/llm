import requests
import json
from typing import Dict, Any, Tuple
from uuid import uuid4
import os

def test_llama_guard_connection(endpoint: str) -> bool:
    """Test if the Llama Guard endpoint is operational"""
    try:
        # Simple test message
        test_payload = {
            "model": "llama-guard3:latest",
            "messages": [
                {"role": "user", "content": "Hello, this is a test message."}
            ],
            "stream": True
        }
        
        response = requests.post(
            endpoint,
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        # Check if we got a valid response
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing Llama Guard connection: {e}")
        return False

def check_prompt_safety(prompt: str, response=None, ollama_endpoint: str = "http://localhost:11434/api/generate") -> Tuple[bool, Dict[str, Any]]:
    """
    Use Llama Guard to check if a prompt is safe.
    Returns a tuple: (is_attack_detected, safety_report)
    """
    try:
        content = prompt
        if response:
            content = f"USER PROMPT: {prompt}\n\nMODEL RESPONSE: {response}"
            
        payload = {
            "model": "llama-guard3:latest",
            "messages": [
                {"role": "user", "content": content}
            ],
            "stream": True
        }
        
        api_response = requests.post(
            ollama_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if api_response.status_code != 200:
            return True, {
                "error": f"Llama Guard API returned status code {api_response.status_code}",
                "is_safe": False,
                "unsafe_categories": ["error"],
                "prompt": prompt
            }
        
        # Parse the Llama Guard response
        resp_json = api_response.json()
        
        # Extract the safety analysis from the response
        guard_results = {
            "raw_response": resp_json,
            "prompt": prompt,
            "response": response,
            "is_safe": True,  # Default to safe
            "unsafe_categories": []
        }
        
        # Parse the content to extract safety categories
        content = resp_json.get("message", {}).get("content", "")
        
        # Look for unsafe keywords in the response
        is_attack_detected = "unsafe" in content.lower()
        if is_attack_detected:
            guard_results["is_safe"] = False
            
            # Extract categories - this parsing would depend on Llama Guard's exact output format
            categories = []
            for line in content.split("\n"):
                line = line.lower()
                if ":" in line and "unsafe" in line:
                    category = line.split(":")[0].strip()
                    categories.append(category)
            
            guard_results["unsafe_categories"] = categories
                
        return is_attack_detected, guard_results
    
    except Exception as e:
        print(f"Error checking safety: {e}")
        return True, {
            "error": str(e),
            "is_safe": False,
            "unsafe_categories": ["error"],
            "prompt": prompt
        }

def print_safety_report(attack_type: str, prompt: str, is_unsafe: bool, guard_results: Dict[str, Any], step: int = 0, max_steps: int = 1) -> None:
    """Print a human-readable summary of the safety report"""
    print(f"\n==== PROMPT SAFETY REPORT ({attack_type}, step {step+1}/{max_steps}) ====")
    print(f"Llama Guard: {'⚠️ UNSAFE' if is_unsafe else '✅ SAFE'}")
    
    if guard_results.get("unsafe_categories"):
        print("\nCategories:")
        for category in guard_results.get("unsafe_categories", []):
            print(f"- {category}")
    
    if "error" in guard_results:
        print(f"\nError: {guard_results['error']}")
    
    print(f"\nPrompt: {prompt[:100]}...")
    print("============================\n")

def print_response_safety_report(response: str, guard_results: Dict[str, Any]) -> None:
    """Print a human-readable summary of the response safety report"""
    print("\n==== RESPONSE SAFETY REPORT ====")
    print(f"Response is {'⚠️ UNSAFE' if not guard_results.get('is_safe', True) else '✅ SAFE'}")
    
    if guard_results.get("unsafe_categories"):
        print("\nCategories:")
        for category in guard_results.get("unsafe_categories", []):
            print(f"- {category}")
    
    if "error" in guard_results:
        print(f"\nError: {guard_results['error']}")
    
    print(f"\nResponse: {response[:100]}...")
    print("==============================\n")

def save_safety_report(attack_type: str, template: str, prompt: str, response: str, 
                      is_attack_detected: bool, llama_guard_results: Dict[str, Any], 
                      response_guard_results: Dict[str, Any] = None, risk_score: int = 0,
                      evaluation: str = "", adapt_attempts: int = 0, 
                      reports_dir: str = "llama_guard_reports") -> str:
    """Save the safety report to a file"""
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Create a filename with timestamp and attack type
    timestamp = uuid4()
    filename = f"{attack_type.replace(' ', '_')}_{timestamp}.json"
    file_path = os.path.join(reports_dir, filename)
    
    # Prepare the report
    report = {
        "timestamp": str(timestamp),
        "attack_type": attack_type,
        "template": template,
        "prompt": prompt,
        "response": response,
        "adaptation_step": adapt_attempts,
        "llama_guard": {
            "is_attack_detected": is_attack_detected,
            "prompt_analysis": llama_guard_results,
            "response_analysis": response_guard_results
        },
        "openai_evaluation": {
            "risk_score": risk_score,
            "evaluation": evaluation
        }
    }
    
    # Save the report
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Safety report saved to {file_path}")
    return file_path