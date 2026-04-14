import os
import subprocess

# Bypass Streamlit's first run prompt
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

# print("Installing Streamlit just in case...", flush=True)
# subprocess.run(["pip", "install", "streamlit"], check=False)

print("Starting Streamlit...", flush=True)
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
print("Streamlit process started in background.", flush=True)
