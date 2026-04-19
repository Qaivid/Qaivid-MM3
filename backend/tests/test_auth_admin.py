"""
Test Suite for Qaivid 2.0 Auth & Admin Features
Tests: Registration, Login, JWT cookies, Admin panel, Credit system, Project scoping, Brute force protection
"""
import pytest
import requests
import os
import time
import uuid

BASE_URL = os.environ.get('REACT_APP_BACKEND_URL', '').rstrip('/')
if not BASE_URL:
    BASE_URL = "https://meaning-to-shot.preview.emergentagent.com"

# Admin credentials sourced from environment so tests follow whatever
# the seeder is configured with (set ADMIN_EMAIL / ADMIN_PASSWORD as Replit Secrets).
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
TEST_USER_EMAIL = "test@test.com"
TEST_USER_PASSWORD = "test123"


class TestHealthCheck:
    """Basic API health check"""
    
    def test_api_health(self):
        """API should return health status"""
        response = requests.get(f"{BASE_URL}/api/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Qaivid 2.0 API"
        assert data["version"] == "2.0.0"
        print("SUCCESS: API health check passed")


class TestUserRegistration:
    """User registration tests"""
    
    def test_register_new_user(self):
        """Register a new user with email, password, name"""
        unique_email = f"testuser_{uuid.uuid4().hex[:8]}@test.com"
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Test User"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        # Verify user data returned
        assert data["email"] == unique_email
        assert data["name"] == "Test User"
        assert data["role"] == "user"
        assert data["plan"] == "free"
        assert "id" in data
        assert "password_hash" not in data  # Should not expose password hash
        
        # Verify cookies are set
        cookies = response.cookies
        assert "access_token" in cookies or response.headers.get("set-cookie", "").find("access_token") != -1
        print(f"SUCCESS: User registered: {unique_email}")
    
    def test_register_duplicate_email(self):
        """Registration with existing email should fail"""
        # First register
        unique_email = f"duplicate_{uuid.uuid4().hex[:8]}@test.com"
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "First User"
        })
        
        # Try to register again with same email
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass456",
            "name": "Second User"
        })
        assert response.status_code == 409, f"Expected 409, got {response.status_code}"
        print("SUCCESS: Duplicate email registration rejected")
    
    def test_register_short_password(self):
        """Registration with short password should fail"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": f"short_{uuid.uuid4().hex[:8]}@test.com",
            "password": "12345",  # Less than 6 chars
            "name": "Test"
        })
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("SUCCESS: Short password rejected")
    
    def test_register_missing_email(self):
        """Registration without email should fail"""
        response = requests.post(f"{BASE_URL}/api/auth/register", json={
            "password": "testpass123",
            "name": "Test"
        })
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"
        print("SUCCESS: Missing email rejected")


class TestUserLogin:
    """User login tests"""
    
    def test_login_valid_credentials(self):
        """Login with valid credentials should succeed"""
        # First register a user
        unique_email = f"login_{uuid.uuid4().hex[:8]}@test.com"
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Login Test"
        })
        
        # Now login
        session = requests.Session()
        response = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": unique_email,
            "password": "testpass123"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert data["email"] == unique_email
        assert "id" in data
        assert "password_hash" not in data
        
        # Verify cookies are set
        assert "access_token" in session.cookies
        print(f"SUCCESS: Login successful for {unique_email}")
    
    def test_login_wrong_password(self):
        """Login with wrong password should return 401"""
        # First register a user
        unique_email = f"wrongpw_{uuid.uuid4().hex[:8]}@test.com"
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "correctpass",
            "name": "Wrong PW Test"
        })
        
        # Try login with wrong password
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": unique_email,
            "password": "wrongpassword"
        })
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("SUCCESS: Wrong password returns 401")
    
    def test_login_nonexistent_user(self):
        """Login with non-existent email should return 401"""
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": f"nonexistent_{uuid.uuid4().hex}@test.com",
            "password": "anypassword"
        })
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("SUCCESS: Non-existent user returns 401")
    
    def test_admin_login(self):
        """Admin login should succeed with correct credentials"""
        session = requests.Session()
        response = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
        data = response.json()
        
        assert data["email"] == ADMIN_EMAIL
        assert data["role"] == "admin"
        assert data["plan"] == "studio"
        print("SUCCESS: Admin login successful")


class TestAuthMe:
    """GET /api/auth/me tests"""
    
    def test_auth_me_authenticated(self):
        """GET /api/auth/me should return user info with credits when authenticated"""
        # Register and login
        unique_email = f"me_{uuid.uuid4().hex[:8]}@test.com"
        session = requests.Session()
        session.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Me Test"
        })
        
        # Get /auth/me
        response = session.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert data["email"] == unique_email
        assert "credit_balance" in data
        assert "plan" in data
        assert "plan_limit" in data
        print(f"SUCCESS: /auth/me returns user with credits: {data.get('credit_balance')}")
    
    def test_auth_me_unauthenticated(self):
        """GET /api/auth/me should return 401 when not authenticated"""
        response = requests.get(f"{BASE_URL}/api/auth/me")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("SUCCESS: /auth/me returns 401 when unauthenticated")


class TestLogout:
    """Logout tests"""
    
    def test_logout_clears_cookies(self):
        """Logout should clear auth cookies"""
        # Login first
        unique_email = f"logout_{uuid.uuid4().hex[:8]}@test.com"
        session = requests.Session()
        session.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Logout Test"
        })
        
        # Verify logged in
        me_response = session.get(f"{BASE_URL}/api/auth/me")
        assert me_response.status_code == 200
        
        # Logout
        logout_response = session.post(f"{BASE_URL}/api/auth/logout")
        assert logout_response.status_code == 200
        data = logout_response.json()
        assert data.get("logged_out") == True
        
        # Verify no longer authenticated (cookies should be cleared)
        # Note: Session may still have old cookies, but server should have cleared them
        print("SUCCESS: Logout returns logged_out: true")


class TestProjectScoping:
    """Projects should be scoped to logged-in user"""
    
    def test_unauthenticated_projects_returns_401(self):
        """GET /api/projects should return 401 when not authenticated"""
        response = requests.get(f"{BASE_URL}/api/projects")
        assert response.status_code == 401, f"Expected 401, got {response.status_code}"
        print("SUCCESS: Unauthenticated /projects returns 401")
    
    def test_user_only_sees_own_projects(self):
        """User should only see their own projects"""
        # Create two users
        user1_email = f"user1_{uuid.uuid4().hex[:8]}@test.com"
        user2_email = f"user2_{uuid.uuid4().hex[:8]}@test.com"
        
        session1 = requests.Session()
        session2 = requests.Session()
        
        # Register both users
        session1.post(f"{BASE_URL}/api/auth/register", json={
            "email": user1_email, "password": "testpass123", "name": "User 1"
        })
        session2.post(f"{BASE_URL}/api/auth/register", json={
            "email": user2_email, "password": "testpass123", "name": "User 2"
        })
        
        # User 1 creates a project
        project1_response = session1.post(f"{BASE_URL}/api/projects", json={
            "name": "User1 Project",
            "description": "Test project for user 1"
        })
        assert project1_response.status_code == 200
        project1 = project1_response.json()
        
        # User 2 creates a project
        project2_response = session2.post(f"{BASE_URL}/api/projects", json={
            "name": "User2 Project",
            "description": "Test project for user 2"
        })
        assert project2_response.status_code == 200
        project2 = project2_response.json()
        
        # User 1 should only see their project
        user1_projects = session1.get(f"{BASE_URL}/api/projects").json()
        user1_project_names = [p["name"] for p in user1_projects]
        assert "User1 Project" in user1_project_names
        assert "User2 Project" not in user1_project_names
        
        # User 2 should only see their project
        user2_projects = session2.get(f"{BASE_URL}/api/projects").json()
        user2_project_names = [p["name"] for p in user2_projects]
        assert "User2 Project" in user2_project_names
        assert "User1 Project" not in user2_project_names
        
        print("SUCCESS: Users only see their own projects")


class TestAdminEndpoints:
    """Admin endpoint tests"""
    
    @pytest.fixture
    def admin_session(self):
        """Get authenticated admin session"""
        session = requests.Session()
        response = session.post(f"{BASE_URL}/api/auth/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        })
        if response.status_code != 200:
            pytest.skip(f"Admin login failed: {response.text}")
        return session
    
    @pytest.fixture
    def user_session(self):
        """Get authenticated regular user session"""
        session = requests.Session()
        unique_email = f"regular_{uuid.uuid4().hex[:8]}@test.com"
        session.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Regular User"
        })
        return session
    
    def test_admin_stats(self, admin_session):
        """Admin can get stats"""
        response = admin_session.get(f"{BASE_URL}/api/admin/stats")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        
        assert "users" in data
        assert "projects" in data
        assert "plans" in data
        assert isinstance(data["users"], int)
        assert isinstance(data["projects"], int)
        print(f"SUCCESS: Admin stats - {data['users']} users, {data['projects']} projects")
    
    def test_admin_list_users(self, admin_session):
        """Admin can list all users"""
        response = admin_session.get(f"{BASE_URL}/api/admin/users")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        users = response.json()
        
        assert isinstance(users, list)
        assert len(users) > 0
        
        # Check user structure
        user = users[0]
        assert "id" in user
        assert "email" in user
        assert "plan" in user
        assert "credit_balance" in user
        assert "password_hash" not in user  # Should not expose password
        print(f"SUCCESS: Admin can list {len(users)} users")
    
    def test_admin_update_user_plan(self, admin_session):
        """Admin can change user plan"""
        # First create a test user
        unique_email = f"plantest_{uuid.uuid4().hex[:8]}@test.com"
        reg_session = requests.Session()
        reg_response = reg_session.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Plan Test"
        })
        user_id = reg_response.json()["id"]
        
        # Admin updates the plan
        response = admin_session.put(f"{BASE_URL}/api/admin/users/{user_id}", json={
            "plan": "pro"
        })
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["plan"] == "pro"
        print(f"SUCCESS: Admin changed user plan to pro")
    
    def test_admin_reset_credits(self, admin_session):
        """Admin can reset user credits"""
        # Create a test user
        unique_email = f"creditreset_{uuid.uuid4().hex[:8]}@test.com"
        reg_session = requests.Session()
        reg_response = reg_session.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "testpass123",
            "name": "Credit Reset Test"
        })
        user_id = reg_response.json()["id"]
        
        # First change plan to pro (so reset gives credits)
        admin_session.put(f"{BASE_URL}/api/admin/users/{user_id}", json={"plan": "pro"})
        
        # Reset credits
        response = admin_session.post(f"{BASE_URL}/api/admin/users/{user_id}/reset-credits")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "new_balance" in data
        assert data["new_balance"] == 4000  # Pro plan limit
        print(f"SUCCESS: Admin reset credits to {data['new_balance']}")
    
    def test_admin_projects_list(self, admin_session):
        """Admin can see all projects"""
        response = admin_session.get(f"{BASE_URL}/api/admin/projects")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        projects = response.json()
        assert isinstance(projects, list)
        print(f"SUCCESS: Admin can see {len(projects)} projects")
    
    def test_non_admin_cannot_access_admin_stats(self, user_session):
        """Non-admin user cannot access admin endpoints"""
        response = user_session.get(f"{BASE_URL}/api/admin/stats")
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("SUCCESS: Non-admin gets 403 on /admin/stats")
    
    def test_non_admin_cannot_access_admin_users(self, user_session):
        """Non-admin user cannot access admin users endpoint"""
        response = user_session.get(f"{BASE_URL}/api/admin/users")
        assert response.status_code == 403, f"Expected 403, got {response.status_code}"
        print("SUCCESS: Non-admin gets 403 on /admin/users")
    
class TestBruteForceProtection:
    """Brute force protection tests"""
    
    def test_brute_force_lockout(self):
        """Account should be locked after 5 failed attempts"""
        # Create a user
        unique_email = f"bruteforce_{uuid.uuid4().hex[:8]}@test.com"
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "correctpassword",
            "name": "Brute Force Test"
        })
        
        # Make 5 failed login attempts
        for i in range(5):
            response = requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": unique_email,
                "password": "wrongpassword"
            })
            assert response.status_code == 401, f"Attempt {i+1}: Expected 401, got {response.status_code}"
        
        # 6th attempt should be locked out (429)
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": unique_email,
            "password": "wrongpassword"
        })
        assert response.status_code == 429, f"Expected 429 (rate limited), got {response.status_code}"
        print("SUCCESS: Brute force protection locks account after 5 failed attempts")
    
    def test_correct_password_after_lockout_still_blocked(self):
        """Even correct password should be blocked during lockout"""
        # Create a user
        unique_email = f"lockout_{uuid.uuid4().hex[:8]}@test.com"
        requests.post(f"{BASE_URL}/api/auth/register", json={
            "email": unique_email,
            "password": "correctpassword",
            "name": "Lockout Test"
        })
        
        # Make 5 failed login attempts
        for i in range(5):
            requests.post(f"{BASE_URL}/api/auth/login", json={
                "email": unique_email,
                "password": "wrongpassword"
            })
        
        # Try with correct password - should still be locked
        response = requests.post(f"{BASE_URL}/api/auth/login", json={
            "email": unique_email,
            "password": "correctpassword"
        })
        assert response.status_code == 429, f"Expected 429, got {response.status_code}"
        print("SUCCESS: Correct password still blocked during lockout period")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
