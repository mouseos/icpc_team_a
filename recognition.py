import face_recognition
import cv2
import numpy as np
import glob
import re
import ctypes
import collections

#enable color for windows
ENABLE_PROCESSED_OUTPUT = 0x0001
ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
MODE = ENABLE_PROCESSED_OUTPUT + ENABLE_WRAP_AT_EOL_OUTPUT + ENABLE_VIRTUAL_TERMINAL_PROCESSING
 
kernel32 = ctypes.windll.kernel32
handle = kernel32.GetStdHandle(-11)
kernel32.SetConsoleMode(handle, MODE)


def extract_file_name(path):
	#Delete all except file name
	#For windows
	name=(re.sub(r".*\\","",path))
	#For unix
	name=(re.sub(r".*/","",name))
	#Delete extension
	name=(re.sub(r"\..*","",name))
	return(name)

print(extract_file_name("path/to/file.png"))

video_capture = cv2.VideoCapture(cv2.CAP_DSHOW)


#Define the path included faces
face_path="./faces"
face_mask_path=face_path+"/trained/with_mask/*"
face_no_mask_path=face_path+"/trained/without_mask/*"
#Collecting each files.
mask_files = glob.glob(face_mask_path)
no_mask_files = glob.glob(face_no_mask_path)

known_face_encodings=[]
known_face_names=[]
#load without mask image
for no_mask_file in no_mask_files:
	print("Loading "+no_mask_file)
	std_id=extract_file_name(no_mask_file)
	try:
		#load face image(without mask)
		known_face_encodings.append(np.load(face_path+"/trained/without_mask/"+std_id+'.npy'))
		known_face_names.append(std_id+"(without mask)")
	except:
		print('\033[31m'+"Can't detect face.Please load another one.")
		#reset default color
		print('\033[39m')
	print(known_face_names)
#load with mask image
for mask_file in mask_files:
	print("Loading "+mask_file)
	std_id=extract_file_name(mask_file)
	
	try:
		#load face image(without mask)
		known_face_encodings.append(np.load(face_path+"/trained/with_mask/"+std_id+'.npy'))
		known_face_names.append(std_id+"(with mask)")
	except:
		print('\033[31m'+"Can't detect face.Please load another one.")
		#reset default color
		print('\033[39m')
	print(known_face_names)
	
	
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
detected_names=[]
scan_cnt=0
send_frames=10;
while True:
	# Grab a single frame of video
	ret, frame = video_capture.read()
	
	# Only process every other frame of video to save time
	if process_this_frame:
		# Resize frame of video to 1/4 size for faster face recognition processing
		small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

		# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
		rgb_small_frame = small_frame[:, :, ::-1]
		
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			# # If a match was found in known_face_encodings, just use the first one.
			# if True in matches:
			#     first_match_index = matches.index(True)
			#     name = known_face_names[first_match_index]

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]

			face_names.append(name)

	process_this_frame = not process_this_frame


	# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Draw a box around the face
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
		
		#Get name
		detected_names.append(name)
		
		scan_cnt+=1
		
		if(scan_cnt==send_frames):
			scan_cnt=0
			detected_names_order=collections.Counter(detected_names)
			print(detected_names_order.most_common())
			detected_names=[]
		
	# Display the resulting image
	cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)#not working aspect ratio is incorrect
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to quit!
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
