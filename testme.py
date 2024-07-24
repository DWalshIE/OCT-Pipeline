import cv2
import pytesseract

# Load Tesseract OCR engine
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to extract text from an image
def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    text = pytesseract.image_to_string(gray)
    return text

# Open a video capture device
cap = cv2.VideoCapture(0)

# Define variables to store student information
student_name = ""
student_number = ""
course_name = ""

# Loop over frames from the video capture device
while True:
    # Capture frame from the video capture device
    ret, frame = cap.read()

    # Extract text from the captured frame
    text = extract_text(frame)

    # Extract student information from the extracted text
    if "Name" in text:
        student_name = text.split("Name")[1].strip()
    if "Student ID" in text:
        student_number = text.split("Student ID")[1].strip()
    if "Course" in text:
        course_name = text.split("Course")[1].strip()

    # Display the captured frame and extracted text in separate windows
    cv2.imshow('Frame', frame)
    cv2.imshow('Text', cv2.putText(frame,text,(0,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA))
    if "studentName" in text:
        student_name2 = text.split("Name")[1].strip()

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save student information to a file
with open('student_info.txt', 'w') as f:
    f.write("Student Name: {}\n".format(student_name))
    f.write("Student Number: {}\n".format(student_number))
    f.write("Course Name: {}\n".format(course_name))

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
