
from fastapi import FastAPI, Request, Response, Depends, status, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes import auth, image
from auth_utils import get_current_user, require_auth, sessions
from translations import get_translations
import uuid
import httpx
import json
import os
import logging
from typing import Optional
from pathlib import Path
import csv
from io import StringIO
from tenacity import retry, stop_after_attempt, wait_fixed
import fcntl
import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Custom Jinja2 filter for translations
def translate(text: str, lang: str = "en"):
    """Translate text using the provided language."""
    translations = get_translations(lang)
    translated = translations.get(text, text)
    logger.debug(f"Translating '{text}' to '{translated}' for lang '{lang}'")
    return translated

templates.env.filters['trans'] = translate

# Satellite data storage
SATELLITE_DATA_DIR = Path("satellite_data")
SATELLITE_DATA_DIR.mkdir(exist_ok=True)

# Include modular routes
app.include_router(auth.router, prefix="/auth")
app.include_router(image.router, prefix="/image")

# Device storage file
DEVICE_FILE = "user_devices.json"

def load_devices():
    """Load user devices from JSON file with error handling."""
    if not os.path.exists(DEVICE_FILE):
        logger.debug(f"No {DEVICE_FILE} found, returning empty dict")
        return {}
    
    try:
        with open(DEVICE_FILE, "r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            if not isinstance(data, dict):
                logger.error(f"Invalid JSON structure in {DEVICE_FILE}, expected dict")
                raise ValueError("Invalid device data format")
            return data
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse {DEVICE_FILE}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load device data")
    except Exception as e:
        logger.error(f"Error reading {DEVICE_FILE}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load device data")

def save_devices(devices):
    """Save user devices to JSON file with file locking."""
    try:
        with open(DEVICE_FILE, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            json.dump(devices, f, indent=2)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except Exception as e:
        logger.error(f"Failed to save devices: {e}")
        raise HTTPException(status_code=500, detail="Failed to save device data")

async def get_geolocation(ip: str):
    """Fetch geolocation for an IP address, with fallback to Tunis, Tunisia."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"https://ipapi.co/{ip}/json/", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            return {
                "lat": data.get("latitude", 36.8065),
                "lon": data.get("longitude", 10.1815),
                "name": f"{data.get('city', 'Unknown')}, {data.get('country_name', 'Unknown')}",
                "is_fallback": False
            }
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        logger.warning(f"Geolocation failed for IP {ip}: {e}")
        return {
            "lat": 36.8065,
            "lon": 10.1815,
            "name": "Tunis, Tunisia",
            "is_fallback": True
        }

async def fetch_fire_data():
    """Fetch fire hotspots from NASA FIRMS API and cache results."""
    cache_file = SATELLITE_DATA_DIR / "fire_data.json"
    
    # Use UTC time for consistent comparison
    now = datetime.datetime.utcnow()
    
    if cache_file.exists():
        mod_time = datetime.datetime.utcfromtimestamp(cache_file.stat().st_mtime)
        if (now - mod_time).seconds < 3600:
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading fire cache: {e}")

    try:
        api_key = "place_holder_api_key" # Replace with your actual API key
        if not api_key:
            logger.error("NASA FIRMS API key not configured")
            return {"type": "FeatureCollection", "features": []}
        
        async with httpx.AsyncClient() as client:
            # Request last 24 hours of data
            today = now.strftime("%Y-%m-%d")
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/world/1/{today}"
            logger.debug(f"Fetching FIRMS data from: {url}")
            
            response = await client.get(url, timeout=20.0)
            response.raise_for_status()
            
            features = []
            csv_data = response.text.splitlines()
            if not csv_data:
                return {"type": "FeatureCollection", "features": []}
                
            reader = csv.DictReader(csv_data)
            for row in reader:
                try:
                    features.append({
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [float(row["longitude"]), float(row["latitude"])]
                        },
                        "properties": {
                            "brightness": float(row["bright_ti4"]),
                            "acq_date": row["acq_date"],
                            "confidence": row["confidence"]
                        }
                    })
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid FIRMS row: {e}")
                    continue
            
            geojson = {"type": "FeatureCollection", "features": features}
            
            with open(cache_file, "w") as f:
                json.dump(geojson, f)
            
            logger.info(f"Fetched {len(features)} fire data points")
            return geojson
            
    except Exception as e:
        logger.error(f"Error fetching fire data: {e}")
        return {"type": "FeatureCollection", "features": []}

async def fetch_co_data(lat: float, lon: float):
    """Fetch real CO data from Open-Meteo API"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Request CO data for the past 24 hours
            url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=carbon_monoxide&past_days=1"
            
            logger.debug(f"Fetching CO data from: {url}")
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            features = []
            
            # Process hourly data
            for i, (time, value) in enumerate(zip(
                data["hourly"]["time"],
                data["hourly"]["carbon_monoxide"]
            )):
                if value is None:
                    continue
                
                # Add slight coordinate variations for better visualization
                offset = 0.50 * (i % 3)  # Creates a small cluster pattern
                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon + offset, lat + offset]
                    },
                    "properties": {
                        "value": round(value, 2),
                        "unit": "µg/m³",
                        "date": time,
                        "source": "Open-Meteo"
                    }
                })
            
            return {
                "type": "FeatureCollection",
                "features": features
            }
            
    except Exception as e:
        logger.error(f"Error fetching CO data: {str(e)}")
        return {
            "type": "FeatureCollection",
            "features": []
        }
    
@app.get("/translations")
async def get_translation(key: str, lang: str = "en"):
    """Return translated text for a given key and language."""
    translation = translate(key, lang)
    logger.debug(f"JS translation: key='{key}', lang='{lang}', result='{translation}'")
    return {"translation": translation}

@app.get("/get_unique_id")
async def get_unique_id(request: Request, current_user: Optional[str] = Depends(require_auth)):
    """Generate and store unique ID for Raspberry Pi with its IP, linked to user."""
    if not current_user:
        logger.error("No authenticated user provided")
        raise HTTPException(status_code=401, detail="Authentication required")
    
    client_ip = request.client.host
    if not client_ip:
        logger.error("Could not determine client IP")
        raise HTTPException(status_code=400, detail="Could not determine client IP")
    
    unique_id = str(uuid.uuid4())
    devices = load_devices()
    if current_user not in devices:
        devices[current_user] = {}
    devices[current_user][unique_id] = client_ip
    save_devices(devices)
    
    logger.debug(f"Generated unique_id={unique_id} for IP={client_ip}, user={current_user}")
    return {"unique_id": unique_id}

@app.get("/view", response_class=HTMLResponse)
async def view_devices(request: Request, current_user: Optional[str] = Depends(require_auth), lang: str = "en"):
    """Display all devices registered by the authenticated user."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    devices = load_devices().get(current_user, {})
    
    response = templates.TemplateResponse("view_devices.html", {
        "request": request,
        "lang": lang,
        "current_user": current_user,
        "devices": devices
    })
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/view/{unique_id}", response_class=HTMLResponse)
async def view_camera(request: Request, unique_id: str, current_user: Optional[str] = Depends(require_auth), lang: str = "en"):
    """Fetch annotated image from Raspberry Pi and display with map and satellite data."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        uuid.UUID(unique_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID format")
    
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    devices = load_devices().get(current_user, {})
    
    if unique_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found or not owned by user")
    
    pi_ip = devices[unique_id]
    image_url = f"/view_image/{unique_id}"
    
    location = await get_geolocation(pi_ip)
    fire_data = await fetch_fire_data()  # Default to Tunis, Tunisia
    co_data = await fetch_co_data(36.8065, 10.1815)  # Default to Tunis, Tunisia
    
    response = templates.TemplateResponse("view.html", {
        "request": request,
        "lang": lang,
        "image_url": image_url,
        "unique_id": unique_id,
        "location": location,
        "fire_data": json.dumps(fire_data),
        "co_data": json.dumps(co_data),
        "current_user": current_user
    })
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/view_image/{unique_id}")
async def view_image(unique_id: str, current_user: Optional[str] = Depends(require_auth)):
    """Fetch and proxy annotated image from Raspberry Pi, use empty response if unavailable."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        uuid.UUID(unique_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid device ID format")
    
    devices = load_devices().get(current_user, {})
    if unique_id not in devices:
        raise HTTPException(status_code=404, detail="Device not found or not owned by user")
    
    pi_ip = devices[unique_id]
    pi_url = f"http://{pi_ip}:5078/capture"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            for attempt in range(3):
                try:
                    response = await client.get(pi_url)
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "")
                    if "image/jpeg" not in content_type:
                        logger.warning(f"Invalid content type from {pi_url}: {content_type}")
                        return Response(content=b"", media_type="image/jpeg")
                    return Response(content=response.content, media_type="image/jpeg")
                except httpx.RequestError as e:
                    logger.warning(f"Attempt {attempt + 1} failed for {pi_url}: {e}")
                    if attempt == 2:
                        return Response(content=b"", media_type="image/jpeg")
    except Exception as e:
        logger.warning(f"Failed to fetch image from {pi_url}: {e}")
        return Response(content=b"", media_type="image/jpeg")

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request, lang: str = "en"):
    """Serve the main index page."""
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Index page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}")
    response = templates.TemplateResponse("index.html", {"request": request, "lang": lang})
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request, lang: str = "en"):
    """Serve the login page."""
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Login page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}")
    response = templates.TemplateResponse("login.html", {"request": request, "lang": lang})
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/register", response_class=HTMLResponse)
async def get_register(request: Request, lang: str = "en"):
    """Serve the registration page."""
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Register page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}")
    response = templates.TemplateResponse("register.html", {"request": request, "lang": lang})
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/upload", response_class=HTMLResponse)
async def get_upload(request: Request, current_user: Optional[str] = Depends(require_auth), lang: str = "en"):
    """Serve the upload page for authenticated users."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Upload page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}, user={current_user}")
    response = templates.TemplateResponse("upload.html", {"request": request, "current_user": current_user, "lang": lang})
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/result/{filename}", response_class=HTMLResponse)
async def get_result(request: Request, filename: str, current_user: Optional[str] = Depends(require_auth), lang: str = "en"):
    """Serve the result page with the specified image filename."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Result page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}, filename={filename}")
    response = templates.TemplateResponse("result.html", {"request": request, "result_image": filename, "current_user": current_user, "lang": lang})
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/profile", response_class=HTMLResponse)
async def get_profile(request: Request, current_user: Optional[str] = Depends(require_auth), lang: str = "en"):
    """Serve the user profile page."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    lang = request.query_params.get("lang", request.cookies.get("lang", lang))
    logger.debug(f"Profile page: query_lang={request.query_params.get('lang')}, cookie_lang={request.cookies.get('lang')}, final_lang={lang}, user={current_user}")
    user = auth.users.get(current_user, {})
    response = templates.TemplateResponse("profile.html", {
        "request": request,
        "current_user": current_user,
        "user": {
            "name": user.get("name", ""),
            "birthday": user.get("birthday", ""),
            "profile_image": user.get("profile_image", None)
        },
        "lang": lang
    })
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

@app.get("/logout", response_class=RedirectResponse)
async def logout(request: Request, response: Response):
    """Log out the user and redirect to the login page."""
    session_id = request.cookies.get("session_id")
    if session_id in sessions:
        del sessions[session_id]
    lang = request.query_params.get("lang", request.cookies.get("lang", "en"))
    response = RedirectResponse(url=f"/login?lang={lang}", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie("session_id", secure=True, samesite="Lax")
    response.set_cookie(key="lang", value=lang, httponly=True, secure=True, samesite="Lax")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4200)
