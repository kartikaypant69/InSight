import os
import pandas as pd
from datetime import datetime

def create_attendance_file(date, attendance_dir="data/attendance"):
    """
    Create a new attendance file for the given date
    
    Args:
        date: date string in format YYYY-MM-DD
        attendance_dir: directory to store attendance files
    """
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
    
    attendance_file = f"{attendance_dir}/{date}.csv"
    
    if not os.path.exists(attendance_file):
        df = pd.DataFrame(columns=["Name", "Time", "Date"])
        df.to_csv(attendance_file, index=False)

def log_attendance(name, attendance_dir="data/attendance"):
    """
    Log attendance in a CSV file
    
    Args:
        name: name of the person
        attendance_dir: directory to store attendance files
        
    Returns:
        success: True if attendance was logged, False if already logged
    """
    today = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    attendance_file = f"{attendance_dir}/{today}.csv"
    
    # Create file if it doesn't exist
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
        
    if not os.path.exists(attendance_file):
        create_attendance_file(today, attendance_dir)
    
    # Read existing attendance
    df = pd.read_csv(attendance_file)
    
    # Check if already marked attendance
    if name in df["Name"].values:
        return False
    
    # Add new attendance
    new_row = {"Name": name, "Time": timestamp, "Date": today}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save updated attendance
    df.to_csv(attendance_file, index=False)
    
    return True

def get_attendance_report(date=None, attendance_dir="data/attendance"):
    """
    Generate attendance report for a specific date or all dates
    
    Args:
        date: date string in format YYYY-MM-DD, or None for all dates
        attendance_dir: directory containing attendance files
        
    Returns:
        DataFrame: attendance report
    """
    if not os.path.exists(attendance_dir):
        os.makedirs(attendance_dir)
        return pd.DataFrame()
    
    if date:
        attendance_file = f"{attendance_dir}/{date}.csv"
        if os.path.exists(attendance_file):
            return pd.read_csv(attendance_file)
        else:
            return pd.DataFrame()
    else:
        # Combine all attendance files
        all_data = []
        for file in os.listdir(attendance_dir):
            if file.endswith(".csv"):
                file_path = f"{attendance_dir}/{file}"
                df = pd.read_csv(file_path)
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()