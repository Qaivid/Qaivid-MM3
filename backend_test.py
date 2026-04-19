#!/usr/bin/env python3
"""
Qaivid 2.0 Backend API Testing
Tests all CRUD operations and pipeline functionality
"""
import requests
import sys
import json
import time
from datetime import datetime

class QaividAPITester:
    def __init__(self, base_url="https://meaning-to-shot.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.project_id = None
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name}")
        else:
            print(f"❌ {name} - {details}")
        
        self.test_results.append({
            "test": name,
            "success": success,
            "details": details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, timeout=30):
        """Run a single API test"""
        url = f"{self.api_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)

            success = response.status_code == expected_status
            details = f"Status: {response.status_code}"
            
            if not success:
                details += f" (expected {expected_status})"
                if response.text:
                    try:
                        error_data = response.json()
                        details += f" - {error_data.get('detail', response.text[:100])}"
                    except:
                        details += f" - {response.text[:100]}"
            
            self.log_test(name, success, details)
            return success, response.json() if success and response.text else {}

        except requests.exceptions.Timeout:
            self.log_test(name, False, "Request timeout")
            return False, {}
        except Exception as e:
            self.log_test(name, False, f"Error: {str(e)}")
            return False, {}

    def test_health_check(self):
        """Test API health"""
        success, response = self.run_test(
            "API Health Check",
            "GET", "", 200
        )
        return success

    def test_culture_packs(self):
        """Test culture packs endpoint"""
        success, response = self.run_test(
            "List Culture Packs",
            "GET", "culture-packs", 200
        )
        if success and isinstance(response, list) and len(response) > 0:
            self.log_test("Culture Packs Data Validation", True, f"Found {len(response)} culture packs")
        elif success:
            self.log_test("Culture Packs Data Validation", False, "Empty culture packs list")
        return success

    def test_project_crud(self):
        """Test project CRUD operations"""
        # Create project
        project_data = {
            "name": f"Test Project {datetime.now().strftime('%H%M%S')}",
            "description": "Test project for API validation",
            "input_mode": "song",
            "language": "auto",
            "culture_pack": "auto"
        }
        
        success, response = self.run_test(
            "Create Project",
            "POST", "projects", 200, project_data
        )
        
        if success and response.get('id'):
            self.project_id = response['id']
            self.log_test("Project ID Retrieved", True, f"ID: {self.project_id}")
        else:
            self.log_test("Project ID Retrieved", False, "No project ID in response")
            return False

        # List projects
        success, response = self.run_test(
            "List Projects",
            "GET", "projects", 200
        )
        
        if success and isinstance(response, list):
            found_project = any(p.get('id') == self.project_id for p in response)
            self.log_test("Project in List", found_project, f"Found {len(response)} projects")
        
        # Get specific project
        success, response = self.run_test(
            "Get Project",
            "GET", f"projects/{self.project_id}", 200
        )
        
        return success

    def test_input_operations(self):
        """Test source input operations"""
        if not self.project_id:
            self.log_test("Input Operations", False, "No project ID available")
            return False

        # Add input
        input_data = {
            "raw_text": """[Verse 1]
Charkha mera rang da ni
Birha da dard sunaave
Ve mahi, pardesiya

[Chorus]
Dil vich tera pyaar hai
Yaad teri aave""",
            "language_hint": "punjabi",
            "culture_hint": "punjabi_rural_lament"
        }
        
        success, response = self.run_test(
            "Add Source Input",
            "POST", f"projects/{self.project_id}/input", 200, input_data
        )
        
        if success:
            # Validate detection results
            if response.get('detected_type') and response.get('line_count'):
                self.log_test("Input Detection Results", True, 
                            f"Type: {response.get('detected_type')}, Lines: {response.get('line_count')}")
            else:
                self.log_test("Input Detection Results", False, "Missing detection data")
        
        # Get input
        success, response = self.run_test(
            "Get Source Input",
            "GET", f"projects/{self.project_id}/input", 200
        )
        
        return success

    def test_interpretation(self):
        """Test AI interpretation (this may take 10-20 seconds)"""
        if not self.project_id:
            self.log_test("Interpretation", False, "No project ID available")
            return False

        print("🔄 Starting AI interpretation (may take 10-20 seconds)...")
        
        success, response = self.run_test(
            "AI Interpretation",
            "POST", f"projects/{self.project_id}/interpret", 200, timeout=60
        )
        
        if success:
            # Validate context packet structure
            required_fields = ['project_id', 'input_type', 'narrative_mode', 'speaker_model', 'world_assumptions']
            missing_fields = [f for f in required_fields if f not in response]
            
            if not missing_fields:
                self.log_test("Context Packet Structure", True, "All required fields present")
            else:
                self.log_test("Context Packet Structure", False, f"Missing: {missing_fields}")
        
        # Get context
        success, response = self.run_test(
            "Get Context",
            "GET", f"projects/{self.project_id}/context", 200
        )
        
        return success

    def test_scenes_pipeline(self):
        """Test scenes building"""
        if not self.project_id:
            self.log_test("Scenes Pipeline", False, "No project ID available")
            return False

        success, response = self.run_test(
            "Build Scenes",
            "POST", f"projects/{self.project_id}/scenes/build", 200
        )
        
        if success and isinstance(response, list):
            self.log_test("Scenes Data Validation", True, f"Built {len(response)} scenes")
        elif success:
            self.log_test("Scenes Data Validation", False, "Invalid scenes response")
        
        # Get scenes
        success, response = self.run_test(
            "Get Scenes",
            "GET", f"projects/{self.project_id}/scenes", 200
        )
        
        return success

    def test_shots_pipeline(self):
        """Test shots building"""
        if not self.project_id:
            self.log_test("Shots Pipeline", False, "No project ID available")
            return False

        success, response = self.run_test(
            "Build Shots",
            "POST", f"projects/{self.project_id}/shots/build", 200
        )
        
        if success and isinstance(response, list):
            self.log_test("Shots Data Validation", True, f"Built {len(response)} shots")
        elif success:
            self.log_test("Shots Data Validation", False, "Invalid shots response")
        
        # Get shots
        success, response = self.run_test(
            "Get Shots",
            "GET", f"projects/{self.project_id}/shots", 200
        )
        
        return success

    def test_prompts_pipeline(self):
        """Test prompts building"""
        if not self.project_id:
            self.log_test("Prompts Pipeline", False, "No project ID available")
            return False

        success, response = self.run_test(
            "Build Prompts",
            "POST", f"projects/{self.project_id}/prompts/build", 200
        )
        
        if success and isinstance(response, list):
            self.log_test("Prompts Data Validation", True, f"Built {len(response)} prompts")
        elif success:
            self.log_test("Prompts Data Validation", False, "Invalid prompts response")
        
        # Get prompts
        success, response = self.run_test(
            "Get Prompts",
            "GET", f"projects/{self.project_id}/prompts", 200
        )
        
        return success

    def test_validation_and_export(self):
        """Test validation and export functionality"""
        if not self.project_id:
            self.log_test("Validation & Export", False, "No project ID available")
            return False

        # Test validation
        success, response = self.run_test(
            "Project Validation",
            "GET", f"projects/{self.project_id}/validate", 200
        )
        
        # Test exports
        export_formats = ["json", "csv", "prompts", "storyboard"]
        for fmt in export_formats:
            success, response = self.run_test(
                f"Export {fmt.upper()}",
                "GET", f"projects/{self.project_id}/export/{fmt}", 200
            )
        
        return True

    def test_models_endpoint(self):
        """Test GET /api/models - should return 11 model configs"""
        success, response = self.run_test(
            "GET /api/models (11 model configs)",
            "GET", "models", 200
        )
        if success and isinstance(response, list):
            if len(response) == 11:
                self.log_test("Models Count Validation", True, f"Found {len(response)} models as expected")
                # Check for specific models
                model_ids = [m.get('id') for m in response]
                expected_models = ['generic', 'midjourney', 'dall-e', 'flux', 'sdxl', 'runway', 'kling', 'wan_2_6', 'veo', 'pika', 'stable-diffusion']
                missing = [m for m in expected_models if m not in model_ids]
                if not missing:
                    self.log_test("Expected Models Present", True, "All 11 expected models found")
                else:
                    self.log_test("Expected Models Present", False, f"Missing: {missing}")
            else:
                self.log_test("Models Count Validation", False, f"Expected 11 models, got {len(response)}")
        return success

    def test_character_crud(self):
        """Test character profile CRUD operations"""
        if not self.project_id:
            self.log_test("Character CRUD", False, "No project ID available")
            return False

        # Test POST /api/projects/{id}/characters
        char_data = {
            "name": "Test Character",
            "role": "protagonist",
            "description": "A test character for API testing",
            "appearance": "tall, dark hair, expressive eyes",
            "age_range": "25-30",
            "wardrobe": "casual modern clothing",
            "emotional_range": "contemplative to passionate"
        }
        
        success, response = self.run_test(
            "POST /api/projects/{id}/characters",
            "POST", f"projects/{self.project_id}/characters", 200, char_data
        )
        
        character_id = None
        if success:
            character_id = response.get('id')
            self.log_test("Character Creation", True, f"Created character: {character_id}")
        else:
            return False

        # Test GET /api/projects/{id}/characters
        success, response = self.run_test(
            "GET /api/projects/{id}/characters",
            "GET", f"projects/{self.project_id}/characters", 200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            self.log_test("Character List", True, f"Retrieved {len(response)} characters")
        else:
            self.log_test("Character List", False, "Failed to retrieve characters")
            return False

        # Test PUT /api/projects/{id}/characters/{charId}
        if character_id:
            update_data = {"description": "Updated test character description"}
            success, response = self.run_test(
                "PUT /api/projects/{id}/characters/{charId}",
                "PUT", f"projects/{self.project_id}/characters/{character_id}", 200, update_data
            )
            
            if success:
                self.log_test("Character Update", True, "Character updated successfully")
            else:
                return False

            # Test DELETE /api/projects/{id}/characters/{charId}
            success, response = self.run_test(
                "DELETE /api/projects/{id}/characters/{charId}",
                "DELETE", f"projects/{self.project_id}/characters/{character_id}", 200
            )
            
            if success:
                self.log_test("Character Delete", True, "Character deleted successfully")
            else:
                return False

        return True

    def test_environment_crud(self):
        """Test environment profile CRUD operations"""
        if not self.project_id:
            self.log_test("Environment CRUD", False, "No project ID available")
            return False

        # Test POST /api/projects/{id}/environments
        env_data = {
            "name": "Test Environment",
            "description": "A test environment for API testing",
            "time_of_day": "golden hour",
            "mood": "serene",
            "visual_details": "soft lighting, natural textures",
            "architecture": "modern minimalist"
        }
        
        success, response = self.run_test(
            "POST /api/projects/{id}/environments",
            "POST", f"projects/{self.project_id}/environments", 200, env_data
        )
        
        environment_id = None
        if success:
            environment_id = response.get('id')
            self.log_test("Environment Creation", True, f"Created environment: {environment_id}")
        else:
            return False

        # Test GET /api/projects/{id}/environments
        success, response = self.run_test(
            "GET /api/projects/{id}/environments",
            "GET", f"projects/{self.project_id}/environments", 200
        )
        
        if success and isinstance(response, list) and len(response) > 0:
            self.log_test("Environment List", True, f"Retrieved {len(response)} environments")
        else:
            self.log_test("Environment List", False, "Failed to retrieve environments")
            return False

        # Test DELETE /api/projects/{id}/environments/{envId}
        if environment_id:
            success, response = self.run_test(
                "DELETE /api/projects/{id}/environments/{envId}",
                "DELETE", f"projects/{self.project_id}/environments/{environment_id}", 200
            )
            
            if success:
                self.log_test("Environment Delete", True, "Environment deleted successfully")
            else:
                return False

        return True

    def test_continuity_endpoint(self):
        """Test continuity analysis endpoint"""
        if not self.project_id:
            self.log_test("Continuity Analysis", False, "No project ID available")
            return False

        # Test GET /api/projects/{id}/continuity
        # This should fail gracefully if no scenes/shots exist, or return empty data
        success, response = self.run_test(
            "GET /api/projects/{id}/continuity",
            "GET", f"projects/{self.project_id}/continuity", 400  # Expected to fail without scenes
        )
        
        if success:
            self.log_test("Continuity Endpoint", True, "Continuity endpoint correctly requires scenes first")
            return True
        else:
            # Try again - maybe it returns 200 with empty data
            success, response = self.run_test(
                "GET /api/projects/{id}/continuity (alternative)",
                "GET", f"projects/{self.project_id}/continuity", 200
            )
            if success:
                self.log_test("Continuity Endpoint", True, "Continuity endpoint accessible")
                return True
        return False

    def test_cleanup(self):
        """Clean up test project"""
        if not self.project_id:
            return True
            
        success, response = self.run_test(
            "Delete Test Project",
            "DELETE", f"projects/{self.project_id}", 200
        )
        
        return success

    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting Qaivid 2.0 Backend API Tests")
        print(f"🌐 Testing against: {self.base_url}")
        print("=" * 60)
        
        # Basic connectivity
        if not self.test_health_check():
            print("❌ API health check failed - stopping tests")
            return False
        
        # Test new iteration 2 features first
        print("\n🆕 Testing Iteration 2 Features:")
        self.test_models_endpoint()
        
        # Culture packs (deterministic)
        self.test_culture_packs()
        
        # Project CRUD
        if not self.test_project_crud():
            print("❌ Project CRUD failed - stopping pipeline tests")
            return False
        
        # Test character and environment CRUD
        print("\n👥 Testing Character & Environment CRUD:")
        self.test_character_crud()
        self.test_environment_crud()
        
        # Input operations
        if not self.test_input_operations():
            print("❌ Input operations failed - stopping pipeline tests")
            return False
        
        # AI interpretation (the big one)
        if not self.test_interpretation():
            print("❌ AI interpretation failed - stopping pipeline tests")
            return False
        
        # Pipeline operations
        print("\n🎬 Testing Pipeline Operations:")
        self.test_scenes_pipeline()
        self.test_shots_pipeline()
        self.test_prompts_pipeline()
        
        # Test continuity after scenes/shots are built
        print("\n🔗 Testing Continuity Analysis:")
        self.test_continuity_endpoint()
        
        # Validation and export
        self.test_validation_and_export()
        
        # Cleanup
        self.test_cleanup()
        
        # Results
        print("=" * 60)
        print(f"📊 Tests completed: {self.tests_passed}/{self.tests_run} passed")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print(f"⚠️  {self.tests_run - self.tests_passed} tests failed")
            return False

def main():
    tester = QaividAPITester()
    success = tester.run_all_tests()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())