import math

def calcVectors(p1, p2):
    x = p1[0] - p2[0]
    y = p1[1] - p2[1]
    mag = math.sqrt(x * x + y * y)
    return (x, y, mag)

def findAngle(v1, v2):
    theta = (v1[0]*v2[0] + v1[1]*v2[1])/(math.sqrt(v1[0]*v1[0] + v1[1]*v1[1]) * math.sqrt(v2[0]*v2[0] + v2[1]*v2[1]))
    return math.degrees(math.acos(theta))

