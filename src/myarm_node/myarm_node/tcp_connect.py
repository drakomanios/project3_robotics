import socket
import json
from pymycobot.myarm import MyArm
import time

# Αρχικοποίηση ρομπότ
myarm = MyArm("/dev/ttyAMA0", 115200)

HOST = "192.168.0.134"
PORT = 5000

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print(f"Server is listening on {HOST}:{PORT}...")

while True:
    print("Waiting for laptop to connect...")
    # Κάνουμε accept ΜΟΝΟ ΕΔΩ. Μόλις συνδεθεί το laptop, προχωράμε στο διάβασμα.
    conn, addr = server.accept()
    print(f"Connected successfully to laptop at: {addr}")
    
    try:
        while True:
            # Διαβάζουμε το πακέτο που έστειλε το laptop
            data = conn.recv(1024)
            
            # Αν το laptop αποσυνδεθεί, το data θα είναι άδειο, οπότε βγαίνουμε
            if not data:
                print("Laptop closed the connection.")
                break
                
            # Αποκωδικοποίηση των δεδομένων
            try:
                cmd = json.loads(data.decode('utf-8'))
                print("Received payload:", cmd) # Αυτό θα σου δείξει τι έφτασε!
                
                if cmd.get("type") == "angles":
                    print(f"Moving joints to: {cmd['angles']} | Speed: {cmd['speed']}")
                    myarm.send_angles(cmd["angles"], cmd["speed"])
                    time.sleep(1.0)
                    
            except json.JSONDecodeError:
                print("Received something, but it wasn't valid JSON.")
                
    except ConnectionResetError:
        print("Laptop disconnected abruptly.")
    finally:
        # Κλείνουμε το κανάλι σωστά και ξαναπάμε στην αρχή του loop για την επόμενη κίνηση
        conn.close()
        print("Done with this session. Going back to listen...")