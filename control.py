import sys
sys.path.append('..')

import serial
import time
import numpy as np

from kinematics import MyArm



class ArmControl:
    def __init__(self) -> None:
        pass
        self.arm = MyArm()
        self.ser = serial.Serial(
                port = '/dev/cu.usbserial-110',
                baudrate = 9600,
                xonxoff = False,
                timeout=2
        )


        time.sleep(2)
        print("Control Initialized! ")
        self.write_angle(self.arm.thetas_tilde.tolist())

    def moveto(self, target:np.array):
        """
        thetas: np.array length 3 containing target xyz
        """
        for obj in self.arm.IK_PGD(target):
            print(obj)
            self.arm.render()
            theta0, theta1, theta2, _ = self.arm.thetas_tilde
            angles = [theta0, theta1, -theta1 - theta2, 0.]
            
            self.write_angle(angles)
        

    def write_angle(self, angles):
        angles = [np.rad2deg(t) for t in angles]
        s = ("{:.1f} " * 4 + '\n').format(*angles)

        self.ser.write(s.encode('utf-8'))
        self.ser.flush()
        s = self.ser.readline()
        # print(s)

    
if __name__ == '__main__':
    control = ArmControl()
    from words import words

    for c in words['sheng']:
        control.moveto(c)

    