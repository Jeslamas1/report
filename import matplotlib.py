from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import serial
import re
import time
import math

# Establish serial communication with the board
ser = serial.Serial('COM6', 9600, timeout=1)  # Change 'COM6' to your USB port

ax = ay = az = 0.0
yaw_mode = False

# Kalman filter parameters
class KalmanFilter:
    def __init__(self, process_variance=1e-5, measurement_variance=0.01):
        self.process_variance = process_variance  # Process noise variance (Q)
        self.measurement_variance = measurement_variance  # Measurement noise variance (R)
        self.estimate_error = 1.0  # Initial estimation error (P)
        self.current_estimate = 0.0  # Initial estimate (x_hat)
        self.last_estimate = 0.0  # Previous estimate (x_hat_prev)
        self.kalman_gain = 0.0  # Initial Kalman gain

    def update(self, measurement):
        # Prediction update
        self.estimate_error += self.process_variance

        # Measurement update
        self.kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_variance)
        self.current_estimate = self.last_estimate + self.kalman_gain * (measurement - self.last_estimate)
        self.estimate_error = (1 - self.kalman_gain) * self.estimate_error

        # Save the current estimate for the next cycle
        self.last_estimate = self.current_estimate

        return self.current_estimate

# Initialize the Kalman filters for pitch and roll
kalman_pitch = KalmanFilter()
kalman_roll = KalmanFilter()

def resize(width, height):
    if height == 0:
        height = 1
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0 * width / height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

def drawText(position, textString):
    font = pygame.font.SysFont("Courier", 18, True)
    textSurface = font.render(textString, True, (255, 255, 255, 255), (0, 0, 0, 255))
    textData = pygame.image.tostring(textSurface, "RGBA", True)
    glRasterPos3d(*position)
    glDrawPixels(textSurface.get_width(), textSurface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, textData)

def draw():
    global ax, ay, az
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    # Display the pitch, roll, and yaw values
    osd_text = f"Pitch: {ay:.2f}, Roll: {ax:.2f}"
    if yaw_mode:
        osd_text += f", Yaw: {az:.2f}"

    drawText((-2, -2, 2), osd_text)

    # Apply rotations in the correct order
    # Note: OpenGL uses a right-hand coordinate system.
    if yaw_mode:
    glRotatef(az, 0.0, 1.0, 0.0)  # Yaw, rotate around y-axis
    glRotatef(ay, 1.0, 0.0, 0.0)  # Pitch, rotate around x-axis
    glRotatef(ax, 0.0, 1.0, 0.0)  # Roll, rotate around z-axis

    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(0.5, 0.1, -0.5)
    glVertex3f(-0.5, 0.1, -0.5)
    glVertex3f(-0.5, 0.1, 0.5)
    glVertex3f(0.5, 0.1, 0.5)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(0.5, -0.1, 0.5)
    glVertex3f(-0.5, -0.1, 0.5)
    glVertex3f(-0.5, -0.1, -0.5)
    glVertex3f(0.5, -0.1, -0.5)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)

    glColor3f(1.0, 1.0, 0.0)
    glVertex3f(1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, -1.0)

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(-1.0, -0.2, 1.0)

    glColor3f(1.0, 0.0, 1.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(1.0, 0.2, 1.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(1.0, -0.2, -1.0)
    glEnd()


# Low-pass filter parameters
alpha = 0.1  # Smoothing factor (0 < alpha < 1)

import time

previous_time = time.time()

def read_data():
    global ax, ay, az, previous_time

    while True:
        # Request multiple data points
        ser.write(b".")
        time.sleep(0.01)  # Short delay to allow STM32 to respond

        # Read up to 5 lines of data (or adjust as needed)
        lines = ser.readlines(5)
        for line in lines:
            line = line.decode().strip()

            if not line or "Invalid command" in line:
                continue

            # Updated regex to match the new data format from the STM32
            match = re.search(r"Gyro \(dps\) X: (-?\d+\.\d+), Y: (-?\d+\.\d+), Z: (-?\d+\.\d+), Accel \(g\) X: (-?\d+\.\d+), Y: (-?\d+\.\d+), Z: (-?\d+\.\d+)", line)
            if match:
                gyro_x = float(match.group(1))
                gyro_y = float(match.group(2))
                gyro_z = float(match.group(3))
                accel_x = float(match.group(4))
                accel_y = float(match.group(5))
                accel_z = float(match.group(6))

                # Calculate delta time (dt) for gyroscope integration
                current_time = time.time()
                dt = current_time - previous_time
                previous_time = current_time

                # Estimate angles using gyroscope data (integration)
                ax += gyro_x * dt
                ay += gyro_y * dt
                az += gyro_z * dt

                # Normalize azimuth angle to stay within -180 to 180 degrees
                if az > 180:
                    az -= 360
                elif az < -180:
                    az += 360

                # Use accelerometer data for a complementary filter
                accel_angle_x = math.degrees(math.atan2(accel_y, accel_z))
                accel_angle_y = math.degrees(math.atan2(accel_x, accel_z))

                # Apply complementary filter to combine accelerometer and gyroscope data
                alpha = 0.98  # Complementary filter coefficient
                ax = alpha * ax + (1 - alpha) * accel_angle_x
                ay = alpha * ay + (1 - alpha) * accel_angle_y
                az = alpha * az + (1 - alpha) * math.degrees(math.atan2(accel_x, accel_y))

                # Print only the elevation (ay) and angle (ax)
                print(f"Elevation: {ay:.2f}, Angle: {ax:.2f}, Azimuth: {az:.2f}")

        # After processing the buffered data, break the loop to continue with the visualization
        break

def main():
    global yaw_mode

    video_flags = OPENGL | DOUBLEBUF

    pygame.init()
    screen = pygame.display.set_mode((800, 600), video_flags)
    pygame.display.set_caption("Press Esc to quit, z toggles yaw mode")
    resize(800, 600)
    init()
    frames = 0
    ticks = pygame.time.get_ticks()
    while True:
        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                ser.close()
                return
            if event.type == KEYDOWN and event.key == K_z:
                yaw_mode = not yaw_mode
                ser.write(b"z")
        read_data()
        draw()

        pygame.display.flip()
        frames += 1

    print("fps:  %d" % ((frames * 1000) / (pygame.time.get_ticks() - ticks)))

if __name__ == '__main__':
    main()
