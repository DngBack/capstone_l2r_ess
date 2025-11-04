# simple_download_server.py - Chạy trên SERVER
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import tarfile

SOURCE_DIR = "/workspace/capstone_l2r_ess/outputs/logits/inaturalist2018"
ARCHIVE_NAME = "logits.tar.gz"
PORT = 8000

# Tạo archive
print("📦 Creating archive...")
with tarfile.open(ARCHIVE_NAME, "w:gz") as tar:
    tar.add(SOURCE_DIR, arcname="logits")
print(f"✅ Created: {ARCHIVE_NAME} ({os.path.getsize(ARCHIVE_NAME) / 1024 / 1024:.2f} MB)")

# Start server
os.chdir(os.path.dirname(os.path.abspath(ARCHIVE_NAME)))
server_ip = os.popen('hostname -I').read().strip().split()[0]

print(f"\n🌐 Server started!")
print(f"📥 Download: http://{server_ip}:{PORT}/{ARCHIVE_NAME}")
print("Press Ctrl+C to stop\n")

httpd = HTTPServer(('', PORT), SimpleHTTPRequestHandler)
httpd.serve_forever()