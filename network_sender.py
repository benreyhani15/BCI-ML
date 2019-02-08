import os
from socket import *

def send_data_to_vr_cpu(data):
    # Set to address that is outputted by VR app
    print("Set VR computer sender_ip address to: " + gethostbyname(gethostname()))
    host = "10.191.148.125"
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    UDPSock.sendto(bytes(data, 'utf-8'), addr)
    UDPSock.close()
