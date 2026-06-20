import os
import sys
import time
import http.server
import socketserver
import webbrowser
import threading

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress request logging to keep the command line clean
        pass

def start_server(port, root_dir):
    try:
        os.chdir(root_dir)
        handler = QuietHandler
        socketserver.TCPServer.allow_reuse_address = True
        with socketserver.TCPServer(("", port), handler) as httpd:
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}", file=sys.stderr)

def get_free_port(start_port=8080):
    import socket
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", port))
                return port
        except OSError:
            port += 1
    return start_port

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Verify the landing page file exists
    landing_page_rel = os.path.join("src", "landing_page.html")
    landing_page_abs = os.path.join(base_dir, landing_page_rel)
    
    if not os.path.exists(landing_page_abs):
        print(f"Error: Could not find '{landing_page_rel}' in {base_dir}")
        print("Please ensure this script is run from the project root directory.")
        sys.exit(1)
        
    port = get_free_port(8080)
    
    # Start server in a daemon thread so it exits when the main thread stops
    server_thread = threading.Thread(target=start_server, args=(port, base_dir), daemon=True)
    server_thread.start()
    
    # Give the thread a moment to initialize the server
    time.sleep(0.3)
    
    url = f"http://localhost:{port}/src/landing_page.html"
    print("=" * 65)
    print(" Interactive demo")
    print("=" * 65)
    print(f" Local Server: {url}")
    print(" Opening in your default web browser...")
    print(" Press Ctrl+C to terminate.")
    print("=" * 65)
    
    webbrowser.open(url)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nTerminated bye!")
        sys.exit(0)

if __name__ == "__main__":
    main()
