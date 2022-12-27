import face_recognition
import cv2
import numpy as np
import glob
import re
import ctypes
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


video_capture = cv2.VideoCapture(0)

#Define the path included faces
face_path="./faces"
face_mask_path=face_path+"/with_mask/*"
face_no_mask_path=face_path+"/without_mask/*"
#Collecting each files.
mask_files = glob.glob(face_mask_path)
no_mask_files = glob.glob(face_no_mask_path)

#load without mask image
for no_mask_file in no_mask_files:
	print("Loading "+no_mask_file)
	std_id=extract_file_name(no_mask_file)
	try:
		#load face image(without mask)
		load_image_tmp=face_recognition.load_image_file(no_mask_file)
		face_encoding_tmp=face_recognition.face_encodings(load_image_tmp)[0]
		#save trained data
		np.save(face_path+"/trained/without_mask/"+std_id+'.npy', face_encoding_tmp)
		#save
		#https://github.com/ageitgey/face_recognition/issues/427
	except:
		print('\033[31m'+"Can't detect face.Please load another one.")
		#reset default color
		print('\033[39m')
#load with mask image
for mask_file in mask_files:
	print("Loading "+mask_file)
	std_id=extract_file_name(mask_file)
	try:
		#load face image(without mask)
		load_image_tmp=face_recognition.load_image_file(mask_file)
		face_encoding_tmp=face_recognition.face_encodings(load_image_tmp)[0]
		#save trained data
		np.save(face_path+"/trained/with_mask/"+std_id+'.npy', face_encoding_tmp)
	except:
		print('\033[31m'+"Can't detect face.Please load another one.")
		#reset default color
		print('\033[39m')
print('Face training is done.')