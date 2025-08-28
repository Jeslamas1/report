import tkinter as tk
from tkinter import messagebox
import serial
import time
import threading
import math
from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *

# Configure the serial connection (adjust the port and baudrate as per your setup)
ser = serial.Serial('COM6', 115200, timeout=1)
time.sleep(2)

# Initialize variables for azimuth and elevation
azimuth = 0
elevation = 0
ax = ay = az = 0.0
yaw_mode = False

# Function to send command to STM32 and receive the response
def send_command(command):
    if ser.is_open:
        try:
            ser.write(command.encode())
            time.sleep(0.1)
            response = ser.readline().decode().strip()
            if response:
                print(f"Raw response: {response}")
                output_label.config(text=f"Board Response: {response}")
                if "Gyro" in response or "Accel" in response:
                    update_angles_from_response(response)
            else:
                output_label.config(text="No response from board.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send command: {str(e)}")
    else:
        messagebox.showerror("Error", "Serial port not open.")

# Function to update azimuth and elevation from STM32 response
def update_angles_from_response(response):
    global azimuth, elevation, ax, ay, az
    try:
        # Split the response to get individual data points
        parts = response.split(', ')
        gyro_data = {"X": 0, "Y": 0, "Z": 0}
        accel_data = {"X": 0, "Y": 0, "Z": 0}

        # Parse each part to extract gyro and accel values
        for part in parts:
            if ': ' in part:
                key, value = part.split(': ')
                value = int(value.strip())

                if 'Gyro X' in key:
                    gyro_data['X'] = value
                elif 'Gyro Y' in key:
                    gyro_data['Y'] = value
                elif 'Gyro Z' in key:
                    gyro_data['Z'] = value
                elif 'Accel X' in key:
                    accel_data['X'] = value
                elif 'Accel Y' in key:
                    accel_data['Y'] = value
                elif 'Accel Z' in key:
                    accel_data['Z'] = value

        # Update IMU orientation data (ax, ay, az)
        ax = gyro_data['X']
        ay = gyro_data['Y']
        az = gyro_data['Z']

        # Calculate azimuth using gyro X and Y (assumes atan2 range from -π to π)
        azimuth = math.atan2(gyro_data['Y'], gyro_data['X']) * (180 / math.pi)
        if azimuth < 0:
            azimuth += 360

        # Normalize accelerometer values to calculate elevation
        accel_magnitude = math.sqrt(accel_data['X']**2 + accel_data['Y']**2 + accel_data['Z']**2)
        if accel_magnitude == 0:
            elevation = 0
        else:
            norm_accel_x = accel_data['X'] / accel_magnitude
            norm_accel_z = accel_data['Z'] / accel_magnitude
            elevation = math.atan2(norm_accel_z, norm_accel_x) * (180 / math.pi)
            if elevation < 0:
                elevation += 180

        # Print the calculated angles for debugging
        print(f"Calculated Azimuth: {azimuth}°, Elevation: {elevation}°")

    except (IndexError, ValueError) as e:
        print(f"Failed to parse azimuth and elevation: {e}")
        messagebox.showerror("Error", "Failed to parse azimuth and elevation from response.")

# Function to continuously read data from STM32
def continuous_read():
    while True:
        if ser.is_open:
            try:
                # Request data from the gyroscope
                ser.write(b".")  # Send a dot to request data
                time.sleep(0.1)
                response = ser.readline().decode().strip()
                if response:
                    print(f"Raw response: {response}")
                    update_angles_from_response(response)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to collect data: {str(e)}")
        time.sleep(0.5)

# Function to initialize OpenGL settings for rendering the 3D cube
def init_gl():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

# Function to draw a cube representing the IMU orientation
def draw_cube():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0, 0.0, -7.0)

    # Apply rotations based on IMU data
    glRotatef(ay, 1.0, 0.0, 0.0)  # Pitch
    glRotatef(-ax, 0.0, 0.0, 1.0)  # Roll
    glRotatef(az, 0.0, 1.0, 0.0)  # Yaw (if enabled)

    # Draw cube
    glBegin(GL_QUADS)
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, -1.0)
    glVertex3f(-1.0, 0.2, 1.0)
    glVertex3f(1.0, 0.2, 1.0)

    glColor3f(1.0, 0.5, 0.0)
    glVertex3f(1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, 1.0)
    glVertex3f(-1.0, -0.2, -1.0)
    glVertex3f(1.0, -0.2, -1.0)

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
    pygame.display.flip()

# Function to run the Pygame OpenGL loop
def run_pygame():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    init_gl()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        draw_cube()
        pygame.display.flip()
        pygame.time.wait(10)

# Initialize the tkinter GUI window
window = tk.Tk()
window.title("Antenna Control Interface")
window.geometry("700x700")

# Create buttons for motor control
cw_button = tk.Button(window, text="Start Clockwise", command=lambda: send_command("start_cw"), width=20, height=2)
cw_button.pack(pady=5)

ccw_button = tk.Button(window, text="Start Counterclockwise", command=lambda: send_command("start_ccw"), width=20, height=2)
ccw_button.pack(pady=5)

stop_button = tk.Button(window, text="Stop Motor", command=lambda: send_command("stop"), width=20, height=2)
stop_button.pack(pady=5)

# Label to display the response from the board
output_label = tk.Label(window, text="Board Response: ", width=40, height=2)
output_label.pack(pady=10)

# Create canvases for azimuth and elevation visualization
canvas_width = 300
canvas_height = 300

# Azimuth Canvas
azimuth_canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
azimuth_canvas.pack(pady=10)

# Elevation Canvas
elevation_canvas = tk.Canvas(window, width=canvas_width, height=canvas_height, bg="white")
elevation_canvas.pack(pady=10)

# Start the continuous reading thread to update azimuth and elevation
threading.Thread(target=continuous_read, daemon=True).start()

# Start the Pygame OpenGL thread
threading.Thread(target=run_pygame, daemon=True).start()

# Start the tkinter event loop
window.mainloop()

# Close the serial connection when the GUI is closed
ser.close()