import tkinter as tk
from tkinter import messagebox
import numpy as np
import os
import cv2
import time
from datetime import datetime
from PIL import Image, ImageTk

# Import modules
from face_utils import *
from attendance_logger import *
from gui_components import *

root = tk.Tk()
root.configure(bg="#1e1e1e")
root.geometry("1900x1800")

if not os.path.exists('data'):
    os.makedirs('data')
if not os.path.exists('data/embeddings'):
    os.makedirs('data/embeddings')
if not os.path.exists('data/attendance'):
    os.makedirs('data/attendance')


#Global variables
cap = None
is_webcam_running = False
registered_users = {}
face_tracking = {}
current_frame = None
registration_embeddings = []

def start_webcam(mode):
    """Start the webcam feed"""
    global cap, is_webcam_running, current_frame
    
    if is_webcam_running:
        return
        
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open webcam")
        return
        
    is_webcam_running = True

    if mode == "Attendance":
        #attendance_status_label.config(text="Running")
        update_attendance_frame()
    elif mode == "Registration":
        #register_status_label.config(text="Camera started")
        update_register_frame()

def stop_webcam():
    """Stop the webcam feed and destroy video frames"""
    global cap, is_webcam_running, face_tracking, attendance_video_label
    """
    # Cancel any pending updates
    if update_id is not None:
        if attendance_video_label.winfo_exists():
            attendance_video_label.after_cancel(update_id)
        if register_video_label.winfo_exists():
            register_video_label.after_cancel(update_id)
        update_id = None
    """
    # Release camera
    is_webcam_running = False
    if cap is not None:
        cap.release()
        cap = None
    
    # Clear video displays
    clear_video_display(attendance_video_label)
    
    # Reset tracking
    face_tracking = {}
    
    # Update status labels
    
    
    # Disable capture button
    reg_face_button.config(state=tk.DISABLED,bg="#3a3a3a", fg="#707070")

def update_attendance_frame():
    """Update the attendance frame with webcam feed and face recognition"""
    global current_frame, face_tracking, attendance_video_label
    
    if not is_webcam_running:
        return
        
    ret, frame = cap.read()
    if not ret:
        stop_webcam()
        messagebox.showerror("Error", "Failed to capture frame from webcam")
        return
        
    current_frame = frame.copy()
    # Detect and recognize faces
    boxes, names, similarities = detect_and_recognize_faces(frame, registered_users)
    
    current_time = time.time()
    
    # Process recognition times and attendance
    recognition_times = []
    
    for i, name in enumerate(names):
        if name != "Unknown":
            if name not in face_tracking:
                face_tracking[name] = {"start_time": current_time, "logged": False}
            
            # Calculate recognition time
            recognition_time = current_time - face_tracking[name]["start_time"]
            recognition_times.append(recognition_time)
            
            # Check if face has been recognized for 5 seconds
            if recognition_time >= 5 and not face_tracking[name]["logged"]:
                # Mark attendance
                success = log_attendance(name)
                if success:
                    face_tracking[name]["logged"] = True
                    #messagebox.showinfo("Attendance", f"Attendance marked for {name}")
        else:
            recognition_times.append(0)
    
    # Draw faces on frame
    frame = draw_faces(frame, boxes, names, similarities, recognition_times)
    
    # Clean up tracking for faces that are no longer visible
    names_to_remove = []
    for tracked_name in face_tracking:
        if tracked_name not in names:
            names_to_remove.append(tracked_name)
    
    for name in names_to_remove:
        if current_time - face_tracking[name]["start_time"] > 10:  # Remove after 10 seconds of absence
            del face_tracking[name]
    
    # Convert to Tkinter-compatible photo image
    img_tk = convert_cv_to_tk(frame)
    
    # Update label
    attendance_video_label.img_tk = img_tk
    attendance_video_label.config(image=img_tk)
    
    # Schedule next update
    attendance_video_label.after(10, update_attendance_frame)


def convert_cv_to_tk(cv_image):
    """
    Convert OpenCV image to Tkinter-compatible photo image
    
    Args:
        cv_image: OpenCV BGR image
        
    Returns:
        tk_image: Tkinter-compatible photo image
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Convert to Tkinter-compatible photo image
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    return tk_image

def clear_video_display(video_label):
    """
    Clear the video display
    
    Args:
        video_label: video display label widget
    """
    # Create a blank image
    blank_image = Image.new('RGB', (640, 480), color=(240, 240, 240))
    blank_photo = ImageTk.PhotoImage(blank_image)
    
    # Update label
    video_label.img_tk = blank_photo
    video_label.config(image=blank_photo)

def update_register_frame():
    """Update the register frame with webcam feed"""
    global current_frame, attendance_video_label
    
    if not is_webcam_running:
        return
        
    ret, frame = cap.read()
    if not ret:
        stop_webcam()
        messagebox.showerror("Error", "Failed to capture frame from webcam")
        return
        
    current_frame = frame.copy()
    
    # Process frame for registration
    processed_frame, can_capture, _ = process_registration_frame(frame, registration_embeddings)
    
    # Enable/disable capture button
    if can_capture:
        reg_face_button.config(state=tk.NORMAL,bg="#005f99",fg="#ffffff")
    else:
        reg_face_button.config(state=tk.DISABLED)
    
    # Convert to Tkinter-compatible photo image
    img_tk = convert_cv_to_tk(processed_frame)
    
    # Update label
    attendance_video_label.img_tk = img_tk
    attendance_video_label.config(image=img_tk)
    
    # Schedule next update
    attendance_video_label.after(10, update_register_frame)

def start_registration():
    """Start the registration process"""
    global registration_embeddings
    
    name = user_reg_text.get().strip()
    
    if not name:
        messagebox.showerror("Error", "Please enter a name")
        return
        
    # Reset progress
    reg_face_button.config(text="Capture Face (0/5)")
    
    # Start webcam if not already running
    if not is_webcam_running:
        start_webcam("Registration")
    
    registration_embeddings = []
    

def capture_face():
    """Capture a face for registration"""
    global registration_embeddings
    
    if not is_webcam_running or current_frame is None:
        messagebox.showerror("Error", "Camera not running")
        return
        
    name = user_reg_text.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name")
        return
        
    # Detect faces
    boxes, _ = detect_faces(current_frame)
    
    if len(boxes) != 1:
        messagebox.showerror("Error", "Exactly one face must be visible")
        return
        
    # Get face embedding
    embedding = get_face_embedding(current_frame, boxes[0])
    
    # Add to embeddings list
    registration_embeddings.append(embedding)
    
    # Update progress
    progress = len(registration_embeddings)
    #register_progress_var.set(progress)
    global reg_face_button
    reg_face_button.config(text=f"Capture Face ({progress}/5)")
    
    # Check if registration is complete
    if progress >= 5:
        # Save embeddings
        save_user_embedding(name, registration_embeddings)
        
        # Update registered users
        registered_users[name] = np.array(registration_embeddings)
        
        # Update status
        messagebox.showinfo("Success", f"Successfully registered {name}")
        
        # Reset form
        user_reg_text.delete(0, tk.END)
        #register_progress_var.set(0)
        reg_face_button.config(text="Capture Face (0/5)")
        registration_embeddings = []
        
        # Update home tab count
        #users_count_label.config(text=str(len(registered_users)))


#Frames
operation_frame = tk.Frame(root,bg = "#2c2c2c", bd = 2,relief="ridge")
user_reg_frame = tk.Frame(root, bg = "#2c2c2c", bd =2,relief="ridge")
camera_cont_frame = tk.Frame(root, bg = "#2c2c2c", bd =2,relief="ridge")
registered_users_frame = tk.Frame(root, bg = "#2c2c2c", bd =2,relief="ridge")
camera_feed_frame = tk.Frame(root, bg = "#000000", bd =2,relief="ridge")

#Labels
op_mode_label = tk.Label(root, text="Operation Mode", bg="#2c2c2c", fg="#ffffff", font=("Calibri", 12, "bold"))
op_mode_label.place(relx= 0.07,rely= 0.09) 

user_reg_label = tk.Label(root, text="User registration", bg="#2c2c2c", fg="#ffffff",font=("Calibri", 12, "bold"))
user_reg_label.place(relx= 0.07,rely= 0.29) 

camera_control_label = tk.Label(root, text="Camera controls", bg="#2c2c2c",fg="#ffffff", font=("Calibri", 12, "bold"))
camera_control_label.place(relx= 0.07,rely= 0.49) 

reg_user_label = tk.Label(root, text="Registered users", bg="#2c2c2c",fg="#ffffff", font=("Calibri", 12, "bold"))
reg_user_label.place(relx= 0.07,rely= 0.69) 

camera_feed_label = tk.Label(root, text="Camera Feed", bg="#2c2c2c",fg="#ffffff", font=("Calibri", 12, "bold"))
camera_feed_label.place(relx= 0.31,rely= 0.09) 

#Placings
operation_frame.place(relx= 0.05, rely= 0.1, relheight= 0.12,relwidth= 0.15)
user_reg_frame.place(relx= 0.05, rely= 0.3, relheight= 0.12,relwidth= 0.15)
camera_cont_frame.place(relx= 0.05, rely= 0.5, relheight= 0.12,relwidth= 0.15)
registered_users_frame.place(relx= 0.05, rely= 0.7, relheight= 0.12,relwidth= 0.15)
camera_feed_frame.place(relx= 0.3, rely= 0.1, relheight= 0.75,relwidth= 0.6)

#Text Field inside user registration label
inside_user_reg_label = tk.Label(user_reg_frame, text="Name:", bg="#2c2c2c",fg="#ffffff", font=("Calibri", 10, "bold"))
inside_user_reg_label.place(relx= 0.02,rely= 0.22) 

user_reg_text = tk.Entry(user_reg_frame,bd=2,bg="#3a3a3a", fg="#e0e0e0", insertbackground="#e0e0e0", relief="flat", font=("Calibri", 12))
user_reg_text.place(relx=0.25, rely=0.22, relwidth=0.7)

#Register face buttion
reg_face_button = tk.Button(user_reg_frame, text="Capture face(0/5)",bd = 2,command=capture_face,bg="#3a3a3a", fg="#707070",
                 relief="flat", borderwidth=0)
reg_face_button.config(state=tk.DISABLED)
reg_face_button.place(relx= 0.05, rely= 0.60, relwidth= 0.9)

# Variable holding value of radio button
current_mode = tk.StringVar(value="Attendance")  # default value
past_mode = current_mode.get

# Radio buttons
attendance_radiobutton = tk.Radiobutton(operation_frame, text="Attendance", variable=current_mode, value="Attendance",font=("Calibri", 12), bg	="#2c2c2c",fg="#e0e0e0",selectcolor="#007acc",activebackground="#2c2c2c",activeforeground="#e0e0e0", borderwidth=0,indicatoron=False)
registration_radiobutton = tk.Radiobutton(operation_frame, text="Registration", variable=current_mode, value="Registration",font=("Calibri", 12),bg	="#2c2c2c",fg="#e0e0e0",selectcolor="#007acc",activebackground="#2c2c2c",activeforeground="#e0e0e0", borderwidth=0,indicatoron=False)

if(current_mode.get != past_mode): stop_webcam()
#place radio buttons
attendance_radiobutton.place(relx=0.1,rely=0.3, relwidth=0.8)
registration_radiobutton.place(relx=0.1,rely=0.6, relwidth=0.8)

#camera control buttons
start_camera_button = tk.Button(camera_cont_frame, text="Start Camera",bd = 2,command=lambda: start_webcam(current_mode.get()),bg="#007acc", fg="#ffffff",
                   activebackground="#005f99", activeforeground="#ffffff", relief="flat", borderwidth=0)
start_camera_button.place(relx= 0.05, rely= 0.25, relwidth= 0.9)
stop_camera_button = tk.Button(camera_cont_frame, text="Stop camera",bd = 2,command=stop_webcam,bg="#007acc", fg="#ffffff",
                   activebackground="#005f99", activeforeground="#ffffff", relief="flat", borderwidth=0)
stop_camera_button.place(relx= 0.05, rely= 0.60, relwidth= 0.9)

attendance_video_label = tk.Label(camera_feed_frame, bg="#000000")
attendance_video_label.place(relx= 0.1,rely= 0.1, relwidth=0.8, relheight= 0.8)

registered_users = load_registered_users()

root.mainloop()