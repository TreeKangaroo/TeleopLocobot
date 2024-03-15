from avp_stream import VisionProStreamer
avp_ip = "10.29.230.57"  # example IP 
s = VisionProStreamer(ip = avp_ip, record = True)

while True:
    r = s.latest
    print(r['head'], r['right_wrist'], r['right_fingers'])