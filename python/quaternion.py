import math
import numpy as na
from assert_utils import assert_sorta_eq, array_equal

class Quaternion:
    def __init__ (self, *args, **margs):
        if len(args) == 4:
            self.q = na.array(args, dtype=na.float64)
        elif len(args) == 1 and len(args[0]) == 4:
            self.q = na.array(args[0][:], dtype=na.float64)
        else:
            raise ValueError("invalid initializer: " + str(args))

        if margs.get("check_norm", True):
            assert_sorta_eq(na.linalg.norm(self.q), 1.0)
            
    @property
    def conjugate(self):
        conjugate = [ self.q[0], -self.q[1], -self.q[2], -self.q[3] ]
        self.conjugate =  Quaternion(conjugate)
        return self.conjugate
    @property
    def w(self):
        return self.q[0]

    @property
    def x(self):
        return self.q[1]

    @property
    def y(self):
        return self.q[2]

    @property
    def z(self):
        return self.q[3]

    def __mul__ (self, other):
        a = self.q
        b = other.q
        return Quaternion (a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
                           a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
                           a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
                           a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0], 
                           check_norm=False)

    def __getitem__ (self, i):
        return self.q[i]

    def __repr__ (self):
        return repr (self.q)
    
    def rotate (self, vector):
        b = Quaternion(0, vector[0], vector[1], vector[2], check_norm=False)
        a = self * b
        b.q[0] = self.q[0]
        b.q[1] = - self.q[1]
        b.q[2] = - self.q[2]
        b.q[3] = - self.q[3]
        c = a * b
        return c[1:]

    @staticmethod
    def null():
        return Quaternion.from_roll_pitch_yaw(0, 0, 0)

    @staticmethod
    def from_yaw (yaw):
        return Quaternion.from_roll_pitch_yaw(0, 0, yaw)
    
    @staticmethod
    def from_roll_pitch_yaw (roll, pitch, yaw):
        halfroll = roll / 2;
        halfpitch = pitch / 2;
        halfyaw = yaw / 2;
        sin_r2 = math.sin (halfroll)
        sin_p2 = math.sin (halfpitch)
        sin_y2 = math.sin (halfyaw)
        cos_r2 = math.cos (halfroll)
        cos_p2 = math.cos (halfpitch)
        cos_y2 = math.cos (halfyaw)
        q =  Quaternion (cos_r2 * cos_p2 * cos_y2 + sin_r2 * sin_p2 * sin_y2,
                         sin_r2 * cos_p2 * cos_y2 - cos_r2 * sin_p2 * sin_y2,
                         cos_r2 * sin_p2 * cos_y2 + sin_r2 * cos_p2 * sin_y2,
                         cos_r2 * cos_p2 * sin_y2 - sin_r2 * sin_p2 * cos_y2)
        return q

    @staticmethod
    def from_axis_angle(*args):
        if len(args) == 3:
            axis_angle = na.array(args, dtype=na.float64)
        elif len(args) == 1:
            axis_angle = na.array(args[0])
        else:
            raise ValueError("Bad arguments: " + `args`)
        theta = na.linalg.norm(axis_angle)
        if theta == 0:
            raise ValueError("bad angle: " + `axis_angle`)

        qxyz = (axis_angle / theta) * math.sin(theta/2)
        
        return Quaternion(math.cos(theta/2), qxyz[0], qxyz[1], qxyz[2])
        
    def to_roll_pitch_yaw (self):
        roll_a = 2 * (self.q[0]*self.q[1] + self.q[2]*self.q[3])
        roll_b = 1 - 2 * (self.q[1]*self.q[1] + self.q[2]*self.q[2])
        roll = math.atan2 (roll_a, roll_b)

        pitch_sin = 2 *  (self.q[0]*self.q[2] - self.q[3]*self.q[1])
        pitch = math.asin (pitch_sin)

        yaw_a = 2 * (self.q[0]*self.q[3] + self.q[1]*self.q[2])
        yaw_b = 1 - 2 * (self.q[2]*self.q[2] + self.q[3]*self.q[3])
        yaw = math.atan2 (yaw_a, yaw_b)
        return roll, pitch, yaw

    def to_rotation_matrix(self):
        x2 = self.x * self.x;
	y2 = self.y * self.y;
	z2 = self.z * self.z;
	xy = self.x * self.y;
	xz = self.x * self.z;
	yz = self.y * self.z;
	wx = self.w * self.x;
	wy = self.w * self.y;
	wz = self.w * self.z;
 
	return na.array(
                [[ 1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy), 0.0,],
                 [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx), 0.0,],
                 [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2), 0.0,],
                 [0.0, 0.0, 0.0, 1.0]])

    def __str__(self):
        return '[%f, %f, %f, %f]' % tuple(self.q)

    def __eq__(self, other):
        if not isinstance(other, Quaternion):
            return False
        return array_equal(self.q, other.q)

    

if __name__ == "__main__":
    q = Quaternion.from_roll_pitch_yaw (0, 0, 2 * math.pi / 16)
    v = [ 1, 0, 0 ]
    print v
    for i in range (16):
        v = q.rotate (v)
        print v
