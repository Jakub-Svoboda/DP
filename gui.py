from PyQt5.uic import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import cv2
import csv
import numpy as np
import datetime
import os
import xml.etree.ElementTree as ET
import re

class Ui(QMainWindow):
	def __init__(self):
		super(Ui, self).__init__()
		
		# Load gui from .ui file
		loadUi('gui/ui.ui', self)

		# Set mouse tracking for tracking events
		self.setMouseTracking(True)

		# Set borderless window style
		self.setWindowFlags(Qt.FramelessWindowHint)

		# Set the custom Top Bar buttons to appropriate functions (minimize, maximize, close)
		self.setButtons()
		
		self.setContentsMargins(0,0,0,0)
		self.scrollAreaWidgetContents.setContentsMargins(0,0,0,0)

		# Load css style sheet for the app
		sshFile="gui/style.sheet"
		with open(sshFile,"r") as fh:
			self.setStyleSheet(fh.read())
			
		self.roundCorners()

		# The top bar widget should move the window when dragged, bind the appropritate function
		self.topBar.mouseMoveEvent = self.moveWindow

		# Click the database button on startup to enable it
		self.databaseClicked()

		# Set the database tabs invisible until a database is loaded
		#self.dbTab.setVisible(False)

		# Display the window
		self.show()
		

	def setButtons(self):
		self.closeButton.clicked.connect(QApplication.quit)
		self.maximizeButton.clicked.connect(self.maximizeClicked)
		self.minimizeButton.clicked.connect(self.showMinimized)
		
		# Set icons for top bar buttons
		self.closeButton.setIcon(QIcon("gui/icons/close.svg"))
		self.maximizeButton.setIcon(QIcon("gui/icons/maximize.svg"))
		self.minimizeButton.setIcon(QIcon("gui/icons/minimize.svg"))

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

		# Connec the capture button
		self.captureButton.clicked.connect(self.captureClicked)

		
	def takeCameraImage(self):
		if hasattr(self, "t"):
			return
		# create the video capture thread
		self.t = VideoThread()
		# connect its signal to the update_image slot
		self.t.change_pixmap_signal.connect(self.update_image)
		# start the thread
		self.t.start()


	def selectImage(self):
		imgPath, _ = QFileDialog.getOpenFileName(self,"Select Image File", "","image (*.jpg *.jpeg *.png *.gif)")
		img = cv2.imread(imgPath)
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)
		self.imageDisplay.setPixmap(qtImg)

	def captureClicked(self):
		if hasattr(self, "t"):
			self.t.requestInterruption()
		print("cap")
		

	#Updates the label with a new opencv image	
	@pyqtSlot(np.ndarray)
	def update_image(self, img):
		h, w, _ = img.shape
		h = h//2
		w = w//2		
		img = cv2.rectangle(img,(w-150,h-190),(w+150,h+190),(0,255,0),3)	
		qtImg = self.convert_cv_qt(img)
		self.imageDisplay.setPixmap(qtImg)
	
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

	def databaseClicked(self):
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border-right: 4px solid rgb(253,170,42);}")
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border: 0px ;}")


	def analysisClicked(self):
		self.analysisButton.setStyleSheet("QPushButton#analysisButton {border-right: 4px solid rgb(253,170,42);}")
		self.databaseButton.setStyleSheet("QPushButton#databaseButton {border: 0px ;}")


	def moveWindow(self, event):
		# If maximized change to normal
		if self.isMaximized():
			self.showNormal()
		# Move the window
		#if event.buttons() == QtCore.Qt.LeftButton:
		self.move(self.pos() + event.globalPos() - self.dragPos)
		self.dragPos = event.globalPos()
		event.accept()	

	# Save the position of the drag
	def mousePressEvent(self, event):
		self.dragPos = event.globalPos()


	def roundCorners(self):
		radius = 10.0
		path = QPainterPath()
		path.addRoundedRect(QRectF(self.rect()), radius, radius)
		mask = QRegion(path.toFillPolygon().toPolygon())
		self.setMask(mask)

	# When the maximize button is clicked, this functions changes the view appropriatelly
	def maximizeClicked(self):
		if self.isMaximized():
			self.showNormal()
		else:
			self.showMaximized()	


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
	window.setAttribute(Qt.WA_NoSystemBackground, True)
	window.setAttribute(Qt.WA_TranslucentBackground, True)
	
	app.exec_()

	
if __name__== "__main__":
	main()


















