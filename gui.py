"""
Author: Jakub Svoboda
Email:  xsvobo0z@stud.fit.vutbr.cz
School: Brno University of Technology

This code handles the overall UI of the app. You will need to have external libraries installed to run 
this without errors, mainly OpenCV, Tensorflow and PyQT5 (see requirements.txt for full dependency list).
"""

from PyQt5.uic import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import cv2
import numpy as np
import pickle
import os

from network import IdentityNetwork
from database import Database


MIN_ACCEPTED_DISTANCE = 0.4			# the minimal angle (radians) between the two embeddings to be accepted as a match

class Ui(QMainWindow):
	""" Mainwindow derived class handling the overall gui functionality.

	Args:
		QMainWindow ([type]): Inherits functionality from PyQT 5 Main Window class.

	Returns:
		int: 0 when properly closed.
	"""

	# Constructor
	def __init__(self):
		""" Constructor for the apps main window. First the widget's set in the gui file are loaded. Next, the CSS is applied to the app. 
			The neural network is initialized, as well as the database that holds the persons embeddings.
			After the setup is completed, the windows .show() method is called to draw the widgets.
		"""
		super(Ui, self).__init__()
		
		# Load gui from .ui file
		loadUi('gui/ui.ui', self)

		# Set mouse tracking for tracking events
		self.setMouseTracking(True)

		# Set the custom Top Bar buttons to appropriate functions (minimize, maximize, close)
		self.setButtons()
		
		self.setContentsMargins(0,0,0,0)
		self.scrollAreaWidgetContents.setContentsMargins(0,0,0,0)

		# Load css style sheet for the app
		sshFile="gui/style.sheet"
		with open(sshFile,"r") as fh:
			self.setStyleSheet(fh.read())

		self.disableWidgets()	
			
		# Initialize MTCNN detector
		self.network = IdentityNetwork()	

		# Click the database button on startup to enable it
		self.databaseClicked()

		# Initialize the databse
		self.initDB()

		# Display the window
		self.show()
		

	def setButtons(self):
		""" Connects control buttons click events to appropriate functions
		"""

		# Set icons for the buttons in the left menu
		self.databaseButton.setIcon(QIcon("gui/icons/database2.png"))
		self.analysisButton.setIcon(QIcon("gui/icons/analysis2.png"))

		# Set icons for save/open/new database buttons.
		self.newDBButton.setIcon(QIcon("gui/icons/add.png"))
		self.saveButton.setIcon(QIcon("gui/icons/download.png"))
		self.loadButton.setIcon(QIcon("gui/icons/upload.png"))

		# Set icons for tab buttons
		self.dbTab.setTabIcon(1 ,QIcon("gui/icons/plus.png"))
		self.dbTab.setTabIcon(0 ,QIcon("gui/icons/loupe.png"))
		
		# Connect appropriate functions to menu buttons
		self.databaseButton.clicked.connect(self.databaseClicked)
		self.analysisButton.clicked.connect(self.analysisClicked)
		
		# Connect functions to add from cam/ add from image buttons
		self.addCameraButton.clicked.connect(self.takeCameraImage)
		self.addImageButton.clicked.connect(self.selectImage)

		# Connect the capture button
		self.captureButton.clicked.connect(self.captureClicked)

		# Connect Load Database / Save Database / New Database buttons to functions
		self.newDBButton.clicked.connect(self.newDB)
		self.loadButton.clicked.connect(self.loadDB)
		self.saveButton.clicked.connect(self.saveDB)

		# Connect the analyze Image/Camera buttons
		self.analyzeImageButton.clicked.connect(self.analyzeImage)
		self.analyzeCameraButton.clicked.connect(self.analyzeCameraImage)

		# Connect the analysis capture button
		self.captureButton2.clicked.connect(self.analyze)


	
	def disableWidgets(self):
		""" Disable widgets which should not be loaded without database being present.
		"""
		self.stackedWidget.setEnabled(False)



	def enableWidgets(self):
		""" Enable widgets when a database is loaded.
		"""
		self.stackedWidget.setEnabled(True)



	def initDB(self):
		""" Initialize a new database object, create a new QTableWidget for the database display and add this to the layout.
		"""
		self.db = None
		self.table = QTableWidget()
		self.verticalLayout_7.addWidget(self.table)



	def analyze(self):
		""" The main image analysis process for matching identities.
			Pulls the image from the window's variable and sends it to the NN.
	 		Then finds the closest face in the DB and sets the result labels appropriatelly.
		"""
		# If camera thred is running, request its iterrupt
		if hasattr(self, "t"):
			self.t.requestInterruption()

		# Pull image from pixmap	
		img = self.analysisImageDisplay.pixmap()
		if img is None: # if there is none yet loaded, no analysis is possible
			return

		# Make sure that the database is not empty
		if self.db.db.shape[0] <= 0:
			print("Empty database")
			self.resultLabel.setText("Database is empty.")
			return	

		# Otherwise, determine the identity of the person in image	
		if hasattr(self, "cvImg"):	
			embedding = self.network.detectFaces(self.cvImg) 	# Get the embedding from the network
			if embedding is None:							 	# If MTCNN has failed detection
				self.resultLabel.setText('No face detected with MTCNN')
				return 
			minId, name, dist = self.db.findFace(embedding)	 	# Get the identity from the database	
			if dist < MIN_ACCEPTED_DISTANCE:
				self.resultLabel.setText('Name: ' + str(name) +'\nID: ' + str(minId) + '\nDistance: ' + str(dist.numpy()))
			else:
				self.resultLabel.setText('No match found in database. \nClosest:' + str(name) +'\nDistance: '+ str(dist.numpy()))	
		else:
			print('No face found')	# print to console when MTCNN fails



	def analyzeImage(self):
		"""	Runs when the analysis of a image file is requested.
		First a dialog is opened for the user to locate it
		and then the image is cut to square and set to the UI widget.
		
		Returns:
			None: When the image read fails (eg camera error), the method ends prematurelly.
		"""
		# if camera thread is already running (left running from camera enrollment) interrupt it
		if hasattr(self, "t"):					
			self.t.requestInterruption() 	
		self.resultLabel.setText('')		# Reset old result labels
		imgPath, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","image (*.jpg *.jpeg *.png *.gif)")	#open file dialog
		if not os.path.isfile(imgPath):
			return
		#img = cv2.imread(imgPath)			# Read image from disk
		img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.cvImg = img
		if img is None:						# Read can fail, just return in this case
			return
		#h, w, _ = img.shape					
		#h = h//2
		#w = w//2		
		#img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)		# convert to numpy type
		self.analysisImageDisplay.setPixmap(qtImg) # set the label widget to the image


	
	@pyqtSlot(np.ndarray)
	def updateAnalysisImage(self, img):
		""" Updates the label (analysisImageDisplay) with a new opencv image.

		Args:
			img (np.mat): a opencv (numpy) image that will be converted to QT and displayed.
		"""
		qtImg = self.convert_cv_qt(img)
		self.analysisImageDisplay.setPixmap(qtImg)


	 
	def loadDB(self):
		""" Whenever the 'load database' button is clicked, this method opens a dialog to locate the file and load the DB pickled object.
		"""
		dbPath, _ = QFileDialog.getOpenFileName(self,"Select Database File", "","Pickle File (*pk)")
		if not os.path.exists(dbPath):
			print('No database file found in path:', dbPath)
			return
		self.db = pickle.load(open(dbPath, "rb" ))
		self.label_2.setText('Database: ' + dbPath)
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))

		# Pull data from the database object
		self.populateTable()
		
		# Enable disabled control widgets
		self.enableWidgets()



	def newDB(self):
		""" Whenever the 'new database' button is clicked, this method opens a dialog to set path and new DB is then created in the place. 
		"""
		# Create new DB object
		self.db = Database()			

		# set info text about DB
		self.label_2.setText('Database: ' + ' unsaved DB')
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))

		# Pull data from the database object
		self.populateTable()

		# Enable disabled control widgets
		self.enableWidgets()	

	

	def populateTable(self, resizeColumns = True):
		""" Pulls the data from the database arrays and fills the table widget.

		Args:
			resizeColumns (bool, optional): If set to True, the colluns will be resized to the maximal content width after the data transfer. Defaults to True.
		"""
		data = self.db.db								# embeddings
		labels = np.expand_dims(self.db.labels,1)		# IDs
		names = np.expand_dims(self.db.names, 1)		# names

		# Set new dimension
		self.table.setRowCount(data.shape[0])
		self.table.setColumnCount(3)					# Name, ID and a remove button

		# Fill the table row by row
		for idx in range(data.shape[0]):
			self.table.setItem(idx, 0, QTableWidgetItem(str(names[idx][0])))	# Add name
			self.table.setItem(idx, 1, QTableWidgetItem(str(labels[idx][0])))	# Add ID
			removeRowButton = QPushButton('')									# Add removal button
			removeRowButton.setIcon(QIcon(os.path.join('gui', 'icons', 'delete.png')))
			removeRowButton.setIconSize(QSize(10, 10))
			removeRowButton.clicked.connect(self.removeRowButtonClicked)
			self.table.setCellWidget(idx, 2, removeRowButton)
		
		# Add a header and resize collumn width
		self.table.setHorizontalHeaderLabels(['Name:', 'ID:', 'Remove:'])
		if resizeColumns:
			self.table.resizeColumnsToContents() 


	
	def removeRowButtonClicked(self):
		""" Removes a row from the identity database.
			First a corresponding button is located (has focus),
			next the appropriate row number is extracted.
			Then the removeId() method of the database is called with this id.
			Lastly, the identity counter text is updated with the new number.
		"""
		button = qApp.focusWidget()
		index = self.table.indexAt(button.pos())
		if index.isValid():
			self.db.removeId(index.row())
		# Repopulate the table widget
		self.populateTable(resizeColumns=False)
		# Update the identity counter
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))


	
	def saveDB(self):
		""" Whenever the 'save database' button is clicked, this function opens a dialog to choose the location and save the DB object there.
		"""
		if self.db is not None:
			dbPath, _ = QFileDialog.getSaveFileName(self,"Create DB File", "","Pickle Files (*pk)")	
			pickle.dump(self.db, open(dbPath, "wb" ) )
			# set info text about DB
			self.label_2.setText('Database: ' + dbPath)
			self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))


	
	def takeCameraImage(self):
		""" Launches a new video Thread at self.t. 
			Then the change_pixmap_signal is connected the update_image() method.
		"""
		#if hasattr(self, "t"):
		#	return
		# create the video capture thread
		self.t = VideoThread()
		# connect its signal to the update_image slot
		self.t.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.t.start()



	def analyzeCameraImage(self):
		""" Launches a new video Thread at self.t. 
			Then the change_pixmap_signal is connected the update_image_noborder() method.
		"""
		self.resultLabel.setText('')
		#if hasattr(self, "t"):
		#	return
		# create the video capture thread
		self.t = VideoThread()
		# connect its signal to the update_image slot
		self.t.change_pixmap_signal.connect(self.update_image_noborder)
		# start the thread
		self.t.start()


	
	def selectImage(self):
		""" Opens a dialog for an image file, loads it and changes the imageDisplay label pixmap.
			A rectangle is drawn in the center of the image to guide users for proper positioning.
		"""
		# if camera thread is already running (left running from camera enrollment) interrupt it
		if hasattr(self, "t"):					
			self.t.requestInterruption() 		
		# open dialog for image selection
		imgPath, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","image (*.jpg *.jpeg *.png *.gif)")
		#img = cv2.imread(imgPath)	# read from disk
		if not os.path.isfile(imgPath):
			return
		img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
		self.cvImg = img
		if img is None:				# if read fails
			return
		qtImg = self.convert_cv_qt(img)										# Convert to numpy
		self.imageDisplay.setPixmap(qtImg)									# Set widget to image


	
	def captureClicked(self):
		""" Runs when the capture button is clicked. The current image is passed through the network and added to the database.
			The table widget that displays the DB is also updated.
		"""
		if hasattr(self, "t"):					# In case when the camera thread is running, request interrupt.
			self.t.requestInterruption()
		# Add image to database
		embeddings = self.network.detectFaces(self.cvImg)			# Forward pass through NN
		self.db.addId(embeddings, name=self.lineEdit.text())		# Add embedding to DB	
		self.populateTable()								
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))	# Update the identity counter label
		self.lineEdit.setText('')									# Reset name input field


		
	@pyqtSlot(np.ndarray)
	def update_image(self, img):
		""" Updates the label with a new opencv image.
			The original cv image (without borders) is preserved for later identification task.
		Args:
			img (np.mat): Image in opencv (numpy) format which is to be displayed.
		"""
		# Save the original cv image for later use
		self.cvImg = img
		# Update the image in enrollment tab
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	# Draw rectangle for the user to position
		qtImg = self.convert_cv_qt(img)
		self.imageDisplay.setPixmap(qtImg)
		


	@pyqtSlot(np.ndarray)
	def update_image_noborder(self, img):
		""" Updates the label with a new opencv image.
		Args:
			img (np.mat): Image in opencv (numpy) format which is to be displayed.
		"""
		# Save the original cv image for later use
		self.cvImg = img
		# update the image in analysis tab
		qtImg2 = self.convert_cv_qt(img)
		self.analysisImageDisplay.setPixmap(qtImg2)
	


	def convert_cv_qt(self, img):
		""" Converts an img from openCV nupy matrix into QTs QImage and then to QPixmap.

		Args:
			img (np.mat): 2D numpy matrix to be converted.

		Returns:
			QPixmap: converted pixmap that can be set to a label.
		"""
		rgbImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgbImage.shape

		# Resize so that on the wider axis the pixels are truncated to a square image
		if w > h:
			rgbImage = rgbImage[::, (w-h)//2 : ((w-h)//2)+h, ::]
			rgbImage = cv2.resize(rgbImage, (300,300), interpolation=cv2.INTER_AREA)
		else:
			rgbImage = rgbImage[(h-w)//2 : ((h-w)//2)+w, ::, ::]
			rgbImage = cv2.resize(rgbImage, (300,300), interpolation=cv2.INTER_AREA)	

		#Convert from an opencv image to QPixmap
		h, w, ch = rgbImage.shape
		bytes_per_line = ch * w
		qtImg = QImage(rgbImage.data, w, h, bytes_per_line, QImage.Format_RGB888)
		return QPixmap.fromImage(qtImg)


	
	def databaseClicked(self):
		""" Reacts to the left menu button clicks. 
			Switches the page from analysis to database.
		"""
		# Set he left menu marker to the appropriate item
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border-right: 4px solid rgb(253,170,42);}")
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border: 0px ;}")
		
		# Switch page to database
		self.stackedWidget.setCurrentIndex(0)
		# Set header text
		self.headerLabel.setText("   Database Manager")


	
	def analysisClicked(self):
		""" Reacts to the left menu button clicks. 
			Switches the page from database to analysis.
		"""
		# Set he left menu marker to the appropriate item
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border-right: 4px solid rgb(253,170,42);}")
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border: 0px ;}")

		# Switch page to analysis
		self.stackedWidget.setCurrentIndex(1)
		# Set header text
		self.headerLabel.setText("   Analysis")



	def closeEvent(self, event):
		"""Request video thread to interrupt when the close button of the app is clicked.

		Args:
			event (QTEvent): Not used for the functionality of this method.
		"""
		if hasattr(self, "t"):
			self.t.requestInterruption()



class VideoThread(QThread):
	""" Thread for managing the webcam image capture.
		Inherits functionality fro the QTs QThread.
	"""
	change_pixmap_signal = pyqtSignal(np.ndarray)

	def run(self):
		""" Checks for the users interrupt request (clicked the Capture button). If detected, releases.
			Otherwise a new image is captured from the webcam and displayed.
		"""
		# Capture from web cam
		cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
		i = 0
		while True:
			# When the user clicks on capture, we request the interrupt and detect it here.	
			if self.isInterruptionRequested():		
				break
			# Otherwise capture image and display it
			ret, cv_img = cap.read()
			if ret:
				self.change_pixmap_signal.emit(cv_img)
		
		# Release the camera and exit thread
		cap.release()
		cv2.destroyAllWindows()
		self.exit(0)		
		
				

def main(args=None):
	""" Run with python 3.6.7 or newer and with libraries specified in requirements.txt.
	"""
	# Allow DPI scaling for high resolution displays
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

	# Create the main app window from an .ui file
	app = QApplication(sys.argv)

	font = qApp.font()
	font.setPixelSize(11)
	app.setFont(font)
	window = Ui()
	
	app.exec_()

	
if __name__== "__main__":
	main()


















