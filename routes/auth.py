from fastapi import APIRouter, HTTPException, status, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from typing import Dict
import secrets
import time
import json
import smtplib
from email.mime.text import MIMEText
from auth_utils import sessions, get_current_user
import os
import uuid
from datetime import datetime, date
import threading

router = APIRouter()

# In-memory storage for OTPs (replace with database for production)
otp_store: Dict[str, Dict] = {}  # email: {code, timestamp, attempts}

# Profile image storage
PROFILE_FOLDER = "static/profiles"
os.makedirs(PROFILE_FOLDER, exist_ok=True)

# Users storage with JSON persistence
USERS_FILE = "users.json"
users_lock = threading.Lock()

def load_users() -> Dict[str, Dict]:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users: Dict[str, Dict]):
    with users_lock:
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=4)

users: Dict[str, Dict] = load_users()  # email: {password, name, birthday, profile_image}

# Load email configuration from JSON file
with open("email_config.json", "r") as f:
    email_config = json.load(f)

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    confirm_password: str
    name: str
    birthday: date

class OTPRequest(BaseModel):
    email: EmailStr
    code: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdateRequest(BaseModel):
    name: str
    birthday: date

def calculate_age(birthday: date) -> int:
    today = datetime.now().date()
    age = today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
    return age

# Send OTP via email
def send_otp_email(email: str, code: str):
    try:
        msg = MIMEText(f"Your OTP is: {code}\nThis code expires in 5 minutes.")
        msg["Subject"] = "Your OTP Code"
        msg["From"] = f"{email_config['sender_name']} <{email_config['sender_email']}>"
        msg["To"] = email

        with smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"]) as server:
            server.starttls()
            server.login(email_config["sender_email"], email_config["sender_password"])
            server.sendmail(email_config["sender_email"], email, msg.as_string())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send OTP email: {str(e)}")

# Generate and send OTP
def generate_otp(email: str):
    if email in otp_store and otp_store[email]["attempts"] >= 5:
        raise HTTPException(status_code=429, detail="Too many attempts. Request a new code.")
    code = str(secrets.randbelow(1000000)).zfill(6)
    otp_store[email] = {"code": code, "timestamp": time.time(), "attempts": 0}
    send_otp_email(email, code)
    return {"message": "OTP sent"}

@router.post("/register")
async def register(data: RegisterRequest):
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    if data.email in users:
        raise HTTPException(status_code=400, detail="Email already registered")
    if calculate_age(data.birthday) < 13:
        raise HTTPException(status_code=400, detail="You must be at least 13 years old to register")
    users[data.email] = {
        "password": data.password,  # Hash in production
        "name": data.name,
        "birthday": data.birthday.isoformat(),
        "profile_image": None
    }
    save_users(users)
    return generate_otp(data.email)

@router.post("/verify-register")
async def verify_register(data: OTPRequest):
    if data.email not in otp_store:
        raise HTTPException(status_code=400, detail="No OTP sent")
    otp_data = otp_store[data.email]
    if otp_data["attempts"] >= 5:
        raise HTTPException(status_code=429, detail="Too many attempts. Request a new code.")
    if time.time() - otp_data["timestamp"] > 300:  # 5-minute expiry
        raise HTTPException(status_code=400, detail="OTP expired")
    if otp_data["code"] != data.code:
        otp_data["attempts"] += 1
        raise HTTPException(status_code=400, detail="Invalid OTP")
    del otp_store[data.email]  # Clear OTP after success
    return {"message": "Registration successful", "redirect": "/login"}

@router.post("/resend-otp")
async def resend_otp(email: EmailStr):
    if email not in users and email not in otp_store:
        raise HTTPException(status_code=400, detail="Email not found")
    return generate_otp(email)

@router.post("/login")
async def login(data: LoginRequest):
    if data.email not in users or users[data.email]["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # Create session
    session_id = secrets.token_hex(16)
    sessions[session_id] = data.email
    response = JSONResponse(content={"message": "Login successful", "redirect": "/upload"})
    response.set_cookie(key="session_id", value=session_id, httponly=True)
    return response

@router.post("/update-profile")
async def update_profile(
    data: ProfileUpdateRequest,
    file: UploadFile = File(None),
    current_user: str = Depends(get_current_user)
):
    if not current_user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if calculate_age(data.birthday) < 13:
        raise HTTPException(status_code=400, detail="You must be at least 13 years old")
    
    user_data = users[current_user]
    user_data["name"] = data.name
    user_data["birthday"] = data.birthday.isoformat()

    if file:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, PNG allowed")
        filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
        filepath = os.path.join(PROFILE_FOLDER, filename)
        with open(filepath, "wb") as f:
            f.write(await file.read())
        # Remove old profile image if exists
        if user_data["profile_image"]:
            old_path = os.path.join(PROFILE_FOLDER, user_data["profile_image"])
            if os.path.exists(old_path):
                os.remove(old_path)
        user_data["profile_image"] = filename

    save_users(users)
    return {"message": "Profile updated successfully"}