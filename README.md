A simple inverse kinematics for 3 DOF robot arm. 

kinematics.py : Forward kinematics with Denavit-Hartenbert frames, simple matplotlib simulation, inverse kinematics with Project Gradient Descent (PGD), inverse kinematics with a home-made Sequential Quadratic Programming (SQP)

control.py : send angles (computed by kinematics) to arduino

MyArm.ino : receive angles from python and write to servo
<p>
  <img width="35%" src="https://github.com/bstars/MyArm/blob/main/IMG_3531.gif">
</p>

