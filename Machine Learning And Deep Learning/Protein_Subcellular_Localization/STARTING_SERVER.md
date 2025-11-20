# Starting the Web Interface

## Quick Start

### Method 1: Using the startup script (Recommended)
```bash
python start_server.py
```

### Method 2: Direct Flask startup
```bash
cd frontend
python app.py
```

## Troubleshooting

### Error: "NetworkError when attempting to fetch resource"

This error means the Flask backend server is not running or not accessible. Follow these steps:

1. **Check if the server is running:**
   - You should see output like: `Running on http://0.0.0.0:5000`
   - If not, start the server using one of the methods above

2. **Check for port conflicts:**
   - Make sure port 5000 is not being used by another application
   - On Windows: `netstat -ano | findstr :5000`
   - On Linux/Mac: `lsof -i :5000`

3. **Check dependencies:**
   - Run: `pip install -r requirements.txt`
   - Make sure all packages install successfully

4. **Check directory paths:**
   - The default paths are: `/mnt/d/5TH_SEM/CELLULAR/input` and `/mnt/d/5TH_SEM/CELLULAR/output`
   - Update these in `frontend/app.py` if your paths are different
   - Or update `config.yaml` with your paths

5. **Check browser console:**
   - Open Developer Tools (F12)
   - Look at the Console tab for detailed error messages
   - Look at the Network tab to see if requests are being sent

### Server is running but still getting NetworkError

1. **Check CORS:**
   - Make sure `flask_cors` is installed: `pip install flask-cors`
   
2. **Check firewall:**
   - Your firewall might be blocking port 5000
   - Try accessing: http://localhost:5000/health
   - You should see: `{"status": "healthy", ...}`

3. **Check for backend errors:**
   - Look at the terminal/console where the Flask server is running
   - Error messages will appear there with [ERROR] prefix

### Image processing takes too long

The processing involves:
- Cellpose segmentation (can take 30-60 seconds per image)
- Superpixel generation
- Graph construction
- Model inference

Be patient and watch the Flask console for progress messages:
```
[INFO] Starting processing for: image.tif
[INFO] Modules imported successfully
[INFO] Image loaded...
[INFO] Segmentation complete...
[INFO] Processing complete
```

## System Requirements

- Python 3.8+
- 8GB+ RAM recommended
- Dependencies listed in requirements.txt
- Optional: CUDA-capable GPU for faster processing

## Configuration

Edit `config.yaml` to customize:
- Input/output directory paths
- Class names
- Image processing parameters
- Segmentation settings
- Model parameters
