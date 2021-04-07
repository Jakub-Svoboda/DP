from PyQt5.uic import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import time
import os

from network import IdentityNetwork
from database import Database



class Ui(QMainWindow):
	def __init__(self):
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
		

	# Connects control buttons to appropriate functions
	def setButtons(self):

		# Set icons for the buttons in the left menu
		self.databaseButton.setIcon(QIcon("gui/icons/database2.png"))
		self.analysisButton.setIcon(QIcon("gui/icons/analysis2.png"))

		# Set icons for save/open database buttons.
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

		# Connect Load Database / Save Database buttons to functions
		self.loadButton.clicked.connect(self.loadDB)
		self.saveButton.clicked.connect(self.saveDB)

		# Connect the analyze Image/Camera buttons
		self.analyzeImageButton.clicked.connect(self.analyzeImage)
		self.analyzeCameraButton.clicked.connect(self.analyzeCameraImage)

		# Connect the analysis capture button
		self.captureButton2.clicked.connect(self.analyze)


	# Disable widgets which should not be loaded without database being present
	def disableWidgets(self):
		self.stackedWidget.setEnabled(False)


	# Enable widgets when a database is loaded
	def enableWidgets(self):
		self.stackedWidget.setEnabled(True)

	# Initialize the database and the display table
	def initDB(self):
		self.db = None
		self.table = QTableView()
		self.verticalLayout_7.addWidget(self.table)


	# The main image analysis process for matching identities
	def analyze(self):
		if hasattr(self, "t"):
			self.t.requestInterruption()
		img = self.analysisImageDisplay.pixmap()
		if img is None:
			return
		if hasattr(self, "cvImg"):
			# Get the embedding from the network	
			embedding = self.network.detectFaces(self.cvImg)
			if embedding is None:
				self.resultLabel.setText('No face detected with MTCNN')
				return 
			# Get the identity from the database
			minId, name, dist = self.db.findFace(embedding)	
			self.resultLabel.setText('Name: ' + str(name) +'\nID: ' + str(minId) + '\nDistance: ' + str(dist.numpy()))
			print(minId, name, dist)
			print('---')
		else:
			print('No face found')


	# Runs when the analysis of a camera feed is requested
	def analyzeImage(self):
		self.resultLabel.setText('')
		imgPath, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","image (*.jpg *.jpeg *.png *.gif)")
		img = cv2.imread(imgPath)
		self.cvImg = img
		if img is None:
			return
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)
		self.analysisImageDisplay.setPixmap(qtImg)


	#Updates the label with a new opencv image	
	@pyqtSlot(np.ndarray)
	def updateAnalysisImage(self, img):
		qtImg = self.convert_cv_qt(img)
		self.analysisImageDisplay.setPixmap(qtImg)


	# Whenever the 'load database' button is clicked, this function opens a dialog to locate the file and load the db object 
	def loadDB(self):
		dbPath, _ = QFileDialog.getOpenFileName(self,"Select Database File", "","Pickle File (*pk)")
		if not os.path.exists(dbPath):
			print('No database file found in path:', dbPath)
			return
		self.db = pickle.load(open(dbPath, "rb" ))
		self.label_2.setText('Database: ' + dbPath)
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))

		# Pull data from the database object
		data = self.db.db
		labels = np.expand_dims(self.db.labels,1)
		names = np.expand_dims(self.db.names, 1)

		#data = np.append(np.expand_dims(self.db.labels,1), data, axis=1)
		self.dbModel = TableModel(names)
		self.table.setModel(self.dbModel)

		#Enable disabled control widgets
		self.enableWidgets()


	# Whenever the 'save database' button is clicked, this function opens a dialog to chose the location and save the db object there
	def saveDB(self):
		if self.db is not None:
			dbPath, _ = QFileDialog.getSaveFileName(self,"Create DB File", "","Pickle Files (*pk)")
			pickle.dump(self.db, open(dbPath, "wb" ) )


	# Launches a new video Thread
	def takeCameraImage(self):
		#if hasattr(self, "t"):
		#	return
		# create the video capture thread
		self.t = VideoThread()
		# connect its signal to the update_image slot
		self.t.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.t.start()


	# Launches a video thread
	def analyzeCameraImage(self):
		self.resultLabel.setText('')
		#if hasattr(self, "t"):
		#	return
		# create the video capture thread
		self.t = VideoThread()
		# connect its signal to the update_image slot
		self.t.change_pixmap_signal.connect(self.update_image_noborder)
		# start the thread
		self.t.start()


	# Opens a dialog for a image file, loads it and changes the imageDisplay label pixmap
	# A rectangle is drawn in the center of the image to guide users for proper positioning
	def selectImage(self):
		imgPath, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","image (*.jpg *.jpeg *.png *.gif)")
		img = cv2.imread(imgPath)
		self.cvImg = img
		if img is None:
			return
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)
		self.imageDisplay.setPixmap(qtImg)


	# When the capture button is clicked, the current image is passed through the network and added to the database
	def captureClicked(self):
		if hasattr(self, "t"):
			self.t.requestInterruption()
		# Add image to database
		embeddings = self.network.detectFaces(self.cvImg)
		self.db.addId(embeddings, name=self.lineEdit.text())
		self.dbModel = TableModel(np.expand_dims(self.db.names, 1))
		self.table.setModel(self.dbModel)
		self.label_3.setText('Identities: ' + str(self.db.labels.shape[0]))


	#Updates the label with a new opencv image	
	@pyqtSlot(np.ndarray)
	def update_image(self, img):
		# Save the original cv image for later use
		self.cvImg = img
		# Update the image in enrollment tab
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)
		self.imageDisplay.setPixmap(qtImg)
		

	#Updates the label with a new opencv image	
	@pyqtSlot(np.ndarray)
	def update_image_noborder(self, img):
		# Save the original cv image for later use
		self.cvImg = img
		# update the image in analysis tab
		qtImg2 = self.convert_cv_qt(img)
		self.analysisImageDisplay.setPixmap(qtImg2)
	

	# Converts an img from openCV nupy matrix into 
	def convert_cv_qt(self, img):
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


	# Reacts on the left menu button clicks. Switches the page from analysis to database
	def databaseClicked(self):
		# Set he left menu marker to the appropriate item
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border-right: 4px solid rgb(253,170,42);}")
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border: 0px ;}")
		
		# Switch page to database
		self.stackedWidget.setCurrentIndex(0)
		# Set header text
		self.headerLabel.setText("   Database Manager")


	# Reacts on the left menu button clicks. Switches the page from database to analysis
	def analysisClicked(self):
		# Set he left menu marker to the appropriate item
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border-right: 4px solid rgb(253,170,42);}")
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border: 0px ;}")

		# Switch page to analysis
		self.stackedWidget.setCurrentIndex(1)
		# Set header text
		self.headerLabel.setText("   Analysis")


	# Request video thread to interrupt when the close button of the app is clicked
	def closeEvent(self, event):
		if hasattr(self, "t"):
			self.t.requestInterruption()


# Displays the database in tabular format
# https://www.learnpyqt.com/tutorials/qtableview-modelviews-numpy-pandas/
class TableModel(QAbstractTableModel):
	def __init__(self, data):
		super(TableModel, self).__init__()
		self._data = data

	def data(self, index, role):
		if role == Qt.DisplayRole:
			value = self._data[index.row(), index.column()]
			return str(value)

	def rowCount(self, index):
		return self._data.shape[0]

	def columnCount(self, index):
		return self._data.shape[1]


# Thread for managing the webcam image capture
class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)

	def run(self):
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
	# Allow DPI scaling for high resolution displays
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	# Create the main app window from an .ui file
	app = QApplication(sys.argv)
	app.setAttribute(Qt.AA_UseHighDpiPixmaps)
	window = Ui()
	
	app.exec_()

	
if __name__== "__main__":
	main()


















