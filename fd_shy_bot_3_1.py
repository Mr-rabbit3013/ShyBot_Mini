# Shy Bot version 3.1

import sensor, time, image, pyb, os
from servo import Servos
from machine import I2C, Pin

BG_UPDATE_FRAMES = 15  #50
BG_UPDATE_BLEND = 0 #128

# fire up servo board
i2c = I2C(sda=Pin('P5'), scl=Pin('P4'))
servo = Servos(i2c, address=0x40, freq=50, min_us=650, max_us=2800, degrees=180)

sensor.reset()

# Sensor settings
sensor.set_contrast(3) #1
sensor.set_gainceiling(16)
sensor.set_auto_whitebal(False)
# HQVGA and GRAYSCALE are the best for face tracking.
sensor.set_framesize(sensor.HQVGA)
sensor.set_pixformat(sensor.GRAYSCALE)
sensor.skip_frames(time = 2000)

# Load Haar Cascade
# By default this will use all stages, lower satges (25)is faster but less accurate.
face_cascade = image.HaarCascade("frontalface", stages=20)
print(face_cascade)

# BG subtraction setup
extra_fb = sensor.alloc_extra_fb(sensor.width(), sensor.height(), sensor.GRAYSCALE)
sensor.skip_frames(time = 2000)
extra_fb.replace(sensor.snapshot())

# FPS clock
clock = time.clock()

def findCenter(targ):
    return (int(targ[0] + (targ[2]/2)), int(targ[1] + (targ[3]/2)))

def translate(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)
    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

servo.position(0,85)    #x
servo.position(1,85)    #y
servo.position(2,85)    #z

frame_count = 0
noTargCount = 0     #counts how long without face target
zBuff = []          #averages the z values for smooth movement

while (True):
    clock.tick()

    # Capture snapshot
    img = sensor.snapshot()

    # Find objects.
    # Note: Lower scale factor scales-down the image more and detects smaller objects.
    # Higher threshold .75results in a higher detection rate, with more false positives.
    #objects = img.find_features(face_cascade, threshold=0.95, scale_factor=1.25) #scale_factor=1.25

    frame_count += 1
    if(frame_count > BG_UPDATE_FRAMES):
        frame_count = 0
        #img.blend(extra_fb, alpha=(256-BG_UPDATE_BLEND))
        extra_fb.replace(img)
    img.difference(extra_fb)
    img.binary([(20,255)])
    #img.erode(4,0)
    img.dilate(2,0)

    #stats = img.get_statistics()

    img.b_and(extra_fb)
    faceRoi = (0,10,240,120)
    img.draw_rectangle(faceRoi)
    objects = img.find_features(face_cascade, threshold=0.75, scale_factor=1.2,roi=faceRoi)

    # Draw objects
    for r in objects:
        img.draw_rectangle(r)
        #print(r)
    #print(len(objects))
    if len(objects) >0 :
        noTargCount = 0

        z = objects[0][2]


        if z < 70:

            zBuff.insert(0,z)
            if len(zBuff) > 10:
                zBuff.pop()
            z_avg = int(sum(zBuff) / len(zBuff))

            x,y = findCenter(objects[0])
            img.draw_cross(x, y, color = (255, 50, 50), size = 10, thickness = 2)
            horz = translate(x,0,240,40,110)
            vert = translate(y,0,160,50,110)
            zed = translate(z_avg,24,69,50,110)
            servo.position(0,horz)
            servo.position(1,vert)
            servo.position(2,zed)

    else:
        noTargCount +=1
        if noTargCount > 100:
            servo.position(2,50)

    # Print FPS.
    # Note: Actual FPS is higher, streaming the FB makes it slower.
    #print(clock.fps())
