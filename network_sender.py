import os
from socket import *

def send_data_to_vr_cpu(data):
    host = "10.191.148.125"
    port = 13000
    addr = (host, port)
    UDPSock = socket(AF_INET, SOCK_DGRAM)
    UDPSock.sendto(bytes(data, 'utf-8'), addr)
    UDPSock.c
    
def get_ip():
    # Set to address that is outputted by VR app
    ip = gethostbyname(gethostname())
    print(ip)
    return ip