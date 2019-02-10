import os
from socket import *

file = 'C:\\Users\\reyhanib\\Documents\\VR\\ReyVR\\VR-master\\Assets\\eeg_buffer.txt'
print("Set host address on EEG CPU to: " + gethostbyname(gethostname()))
host = ""
sender_ip = "10.191.148.214"
port = 13000
buf = 1024
addr = (host, port)
UDPSock = socket(AF_INET, SOCK_DGRAM)
UDPSock.bind(addr)
listening = True
while listening:
    (data, addr) = UDPSock.recvfrom(buf)
    if addr[0] == sender_ip:
        value = data.decode('utf-8')
        print("Received message: " + value)
        #wr = open(file, 'w')
        #wr.write(value)
        if (data.decode('utf-8')) == 'exit':
            listening = False
UDPSock.close()          