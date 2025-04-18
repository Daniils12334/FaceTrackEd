FaceTrackEd

FaceTrackEd is an advanced student attendance tracking and statistical analysis system built with Python, utilizing face_recognition, OpenCV, and Pandas. The application automatically detects students from images or videos and collects data on their attendance frequency.
🔍 Problem Analysis
Problem Description

Traditional student attendance tracking (e.g., using a physical roll-call or manually marking attendance) is slow, prone to human error, and time-consuming. Automated face recognition allows for faster processes, reduces errors, and collects additional statistical data.
Target Audience

    School administration and teachers

    Technology departments or faculties

    Programmers looking to expand their knowledge in computer vision

Analysis of Existing Solutions
Name	Description	Pros	Cons
OpenCV Attendance System (GitHub)	Simple script for attendance tracking	Easy to understand, good base	No statistics, no verification
Commercial solutions (FaceFirst, Trueface)	Professional systems	Accurate, secure	Paid, no open-source, non-customizable
DIY CSV + Camera	Basic face detection	Simple	No recognition, only detection

🧩 Design
Functional Requirements

    Recognize student faces from images or videos.

    Store student ID, name, and face encoding in a database (CSV).

    Log each appearance with date and time.

    Alert when a face is not recognized (unknown student).

    Allow adding a new student to the database.

Non-Functional Requirements

    The application should run via the console (CLI).

    Operation should be possible without internet access.

    Should work with image or video files (not necessarily in real-time).

    The user interface should be simple and intuitive.

    Data storage should be secure (no third-party access).

🗓️ Planning – Task List

    Create a Student class with fields: id, name, and encoding.

    Implement CSV database reading and writing.

    Implement face recognition with face_recognition.

    Log events with date and time.

    Develop a CLI menu (recognition / add / statistics).

🎥 Solution Presentation

    pass

💻 Technologies

    Python 3.x

    face_recognition

    OpenCV

    Pandas

📁 Project Structure

    FaceTrackEd/
    ├── main.py                 
    ├── requirements.txt              
    ├── README.md                    
    ├── config/
    │   └── settings.json       
    │   └── settings.py          
    ├── data/
    │   ├── students.csv         
    │   ├── log.csv                  
    │   └── faces/              
    ├── app/
    │   ├── __init__.py
    │   ├── database/
    │   │   ├── __init__.py
    │   │   ├── db.py                
    │   ├── face/
    │   │   ├── __init__.py
    │   │   ├── recognition.py        
    │   ├── stats/
    │   │   ├── __init__.py
    │   │   ├── analytics.py         
    │   ├── utils/
    │   │   ├── __init__.py
    │   │   ├── helpers.py            
    │   └── core/
    │       ├── __init__.py
    │       ├── app.py                

File Explanations:

    main.py – Starts the application, calls FaceTrackApp.run().

    settings.json – Stores paths, camera settings, date format, etc.

    db.py – Manages loading/saving students and logs.

    recognition.py – Encodes and recognizes faces from the camera.

    analytics.py – Processes and displays statistics.

    app.py – Integrates everything into one application.

    helpers.py – Contains utility functions: timestamps, file path operations, etc.

✅ To-Do
🔹 Project Start

Create GitHub repository: FaceTrackEd

Create README.md with problem description, goals, and plan.

    Prepare test images (student faces).

🔹 Data Structure and Class

Create Student class with fields: id, name, encoding.

Create students.csv file to store data.

    Add the ability to save/load the encoding list from CSV.

🔹 Face Recognition Functionality

Load image or video.

Recognize faces in the image with face_recognition.

Compare with existing faces in the database.

    Alert if face is unrecognized – offer to add.

🔹 Statistics Functions

Save each recognition in log.csv with time and student ID.

Retrieve attendance frequency statistics.

    Create graphs with Matplotlib (optional feature).

🔹 User Interface (CLI)

Create a simple menu in the terminal:

✅ Face recognition

➕ Add new student

    📊 View statistics

    Validate input data (e.g., name input).

🔹 Testing

Test with multiple faces and images.

Simulate errors (e.g., missing encoding).

    Test CSV file corruption and recovery.

