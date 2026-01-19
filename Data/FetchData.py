import serial
import datetime

Serial = serial.Serial("COM37", 115200)

while True:
    data = Serial.readline()
    if(data.startswith(b"[Logs] ")):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        newdata = data.decode('utf-8').strip()
        newdata = timestamp + "," + newdata
        newdata = newdata.replace(" ", "")
        newdata = newdata.replace("[Logs]", "")
        
        actualdata = newdata.split(",")
        with open("Data/logs.csv", "a") as file:
            file.write(newdata + "\n")
        print(actualdata)

