import sys
import os
import traceback
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QHBoxLayout,  QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QProgressBar
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMenuBar, QMenu, QAction, QMessageBox
from PyQt5.QtWidgets import QGraphicsDropShadowEffect,QDialog
from PyQt5.QtGui import QColor, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import ListedColormap
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import imageio.v2 as imageio
from mymodel import Unet, ConvolutionUnit, DownSamplingBlock, UpSamplingBlock
import webbrowser


dir_path = os.path.dirname(__file__)
model_name = "modelFile.h5"  #  model file name
model_path = os.path.join(dir_path, model_name)
model = Unet(
    encoder_channels=[1, 64, 128, 256, 512, 1024],
    decoder_channels=[1024, 512, 256, 128, 64])

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

class ImageSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None  # Initialize the attribute
        self.current_mask_path = None # Initialize the attribute for mask
        self.title = 'Image Segmentation Application'
        self.left, self.top, self.width, self.height = 100, 100, 1000, 800  # Updated dimensions
        self.initUI()

        
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.setStyleSheet("background-color: rgb(54, 56, 71);")
        #Windw icon
        script_dir = os.path.dirname(__file__)
        logo_icon_path = os.path.join(script_dir, 'logo.png')  # Replace 'logo.png' with your logo file
        self.setWindowIcon(QIcon(logo_icon_path))
        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(400, 350, 200, 20)  # Adjust these values to position it over the plot
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(True)

        # Menu Bar
        menu_bar = self.menuBar()

        menu_bar.setStyleSheet("""
            QMenuBar {
                background-color: white;
                color: black;
            }
            QMenuBar::item:selected { /* when selected */
                background-color: #e8e8e8;
            }
            QMenu {
                background-color: white;
                color: black;
            }
            QMenu::item:selected {
                background-color: #e8e8e8;
            }
        """)

        # Drop Shadow Effect
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(0, 0, 0, 60))
        shadow.setOffset(2, 2)
        menu_bar.setGraphicsEffect(shadow)

        file_menu = menu_bar.addMenu('File')

        # Save Action
        save_action = QAction('Save Analysis Image', self)
        save_action.triggered.connect(self.saveData)
        file_menu.addAction(save_action)

        # Contact Menu
        contact_menu = self.menuBar().addMenu('Contact')

        # Facebook Action
        facebook_action = QAction('Facebook', self)
        facebook_action.triggered.connect(lambda: self.open_url('https://www.facebook.com/deb.1337'))
        contact_menu.addAction(facebook_action)

        # GitHub Action
        github_action = QAction('GitHub', self)
        github_action.triggered.connect(lambda: self.open_url('https://github.com/definito'))
        contact_menu.addAction(github_action)

        # Email Action
        email_action = QAction('Email', self)
        email_action.triggered.connect(lambda: self.copy_email_to_clipboard('debwa.web@gmail.com'))
        contact_menu.addAction(email_action)






        # Header layout
        header_layout = QHBoxLayout()

        # Logo label
        logo_label = QLabel(self)
        script_dir = os.path.dirname(__file__)
        image_name = "logo-2.png"  # Replace with your image file name
        image_path = os.path.join(script_dir, image_name)
        logo_pixmap = QtGui.QPixmap(image_path)
        
        if logo_pixmap.isNull():
            print(f"Failed to load logo from '{image_path}'. Check file path and format.")
        else:
            logo_label.setPixmap(logo_pixmap)
            logo_label.setScaledContents(True)
            logo_label.setFixedSize(300, 100)  # Adjust size
            header_layout.addWidget(logo_label)

        # Insert spacer item to push text to the middle/right
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        header_layout.addSpacerItem(spacer)
        # header_layout.addStretch(5)
        # Welcome text label
        text_label = QLabel("Welcome to Lung Analysis!", self)
        text_label.setStyleSheet("font: 75 28pt 'Adobe Caslon Pro'; color: white;")
        header_layout.addWidget(text_label)

        # Widget to apply the header layout
        header_widget = QWidget()
        header_widget.setLayout(header_layout)
        header_widget.setMaximumHeight(100)  # Set maximum height for the header

        # Main layout
        main_layout = QVBoxLayout(self.centralWidget())  # Set the layout on the central widget

        # Add header widget to main layout
        main_layout.addWidget(header_widget)

        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)
        main_layout.addWidget(self.progress_bar)  # Add progress bar to main layout

        # Buttons
        open_btn = QPushButton('Open Image', self)
        open_btn.setStyleSheet("font-size: 14px; color: white; background-color: rgb(33, 150, 243);")
        open_btn.setFixedSize(120, 40)
        open_btn.clicked.connect(self.openFileNameDialog)

        open_mask_btn = QPushButton('Open Mask', self)
        open_mask_btn.setStyleSheet("font-size: 14px; color: white; background-color: rgb(33, 150, 243);")
        open_mask_btn.setFixedSize(120, 40)
        open_mask_btn.clicked.connect(self.openMaskFileDialog)

        segment_btn = QPushButton('Run Segmentation', self)
        segment_btn.setStyleSheet("font-size: 16px; color: white; background-color: rgb(255, 85, 0);")
        segment_btn.setFixedSize(150, 50)
        segment_btn.clicked.connect(self.segmentImage)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(open_btn)
        button_layout.addWidget(open_mask_btn)
        button_layout.addWidget(segment_btn)

        # Add button layout to main layout
        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)


    def open_url(self, url):
        webbrowser.open(url)

    def copy_email_to_clipboard(self, email):
        clipboard = QApplication.clipboard()
        clipboard.setText(email)
        # self.statusBar.showMessage("Email ID copied to clipboard.", 5000) 
       # Create and style the QMessageBox
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("Email Copied")
        msgBox.setText("Email ID has been copied to clipboard.")
        msgBox.setStyleSheet("""
            QMessageBox {
                background-color: white;
            }
            QMessageBox QLabel {
                background-color: white;
            }
            QMessageBox QPushButton {
                background-color: white;

            }
        """)
        msgBox.exec_()
        

    def saveData(self):
        options = QFileDialog.Options()
        save_path = QFileDialog.getSaveFileName(self, "Save File", "", "Image Files (*.png *.jpg *.jpeg);;All Files (*)", options=options)
        if save_path[0]:
            # Save the current figure
            self.saveAnalysisData(save_path[0])

    def saveAnalysisData(self, path):
        # Save the current figure with subplots
        self.figure.savefig(path + '_analysis.png')

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "TIFF Files (*.tiff *.tif);;All Files (*)", options=options)
        if fileName:
            self.current_image_path = fileName  # Update the path
            self.displayImage()  # Update the display

    def openMaskFileDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Mask", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)", options=options)
        if fileName:
            self.current_mask_path = fileName  # Update the mask path
            self.displayImage()  # Update the display
    

    def displayImage(self):
        self.figure.clear()

        # Load and display the original image
        if self.current_image_path:
            original_image = imageio.imread(self.current_image_path)
            ax1 = self.figure.add_subplot(121)
            ax1.imshow(original_image, cmap='gray')
            ax1.axis('off')
            ax1.set_title("Original Image")
        else: 
            # If no image path is set, show placeholder text
            ax1 = self.figure.add_subplot(121)
            ax1.text(0.5, 0.5, 'Load Image Here', ha='center', va='center', fontsize=12)
            ax1.axis('off')
        
        # Load and display the mask if available
        if self.current_mask_path:
            mask_image = imageio.imread(self.current_mask_path)
            ax2 = self.figure.add_subplot(122)
            ax2.imshow(mask_image, cmap='gray')
            ax2.axis('off')
            ax2.set_title("Mask Image")
        else:
            # If no mask path is set, show placeholder text
            ax2 = self.figure.add_subplot(122)
            ax2.text(0.5, 0.5, 'Load Mask Here', ha='center', va='center', fontsize=12)
            ax2.axis('off')

        self.canvas.draw()



    def preprocess_image(self):
        image_path = self.current_image_path
        image = imageio.imread(image_path)

        if len(image.shape) == 2:
            # Grayscale image, no need to convert to grayscale
            image = image[:, :, np.newaxis]  # Add a third channel
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Color image, convert to grayscale
            image = np.mean(image, axis=2)

        # Resize, normalize, etc.
        transform = transforms.Compose([
            transforms.ToTensor(),
            # Add other transformations as used during training
        ])
        return transform(image).unsqueeze(0)  # Add batch dimension

# Run the model on the input image and capture the predicted mask and the prediction
    def segmentImage(self):
        if self.current_image_path:
            self.progress_bar.setVisible(True)  # Show the progress bar
            self.progress_bar.setValue(0)  # Reset progress to 0%

            input_image = self.preprocess_image()
            # Update the progress bar, e.g., at 50%
            self.progress_bar.setValue(50)

            pred, predictedMask = self.run_model(input_image)  # Capture the predicted mask
            # Update the progress bar to 100% on completion
            self.progress_bar.setValue(100)

            self.displaySegmentation(pred, predictedMask)  # Pass it to the display method
            self.progress_bar.setVisible(False)  # Hide the progress bar after completion
    
    def run_model(self, image_tensor):
        with torch.no_grad():
            # # Ensure the input tensor is of type torch.float32
            # image_array = (image_tensor.numpy().astype("float32") - np.mean(image_tensor)) / np.std(image_tensor)

            # # Convert the normalized NumPy array back to a PyTorch tensor
            # # img = torch.from_numpy(im[None, None])
            # image_tensor = torch.from_numpy(image_array[None, None]).to(torch.float32)
                    # Ensure the input tensor is of type torch.float32
            image_tensor = image_tensor.to(torch.float32)

            # Calculate the mean and std along specific dimensions (axis=2 and axis=3 for a 4D tensor)
            mean = torch.mean(image_tensor, dim=(2, 3), keepdim=True)
            std = torch.std(image_tensor, dim=(2, 3), keepdim=True)

            # Normalize the image tensor
            image_tensor = (image_tensor - mean) / (std + 1e-8)  # Adding a small value to avoid division by zero



            # Pass the input tensor through the model
            output = model(image_tensor)

            # # Convert the bias of the convolutional layers to torch.float32
            # for layer in model.modules():
            #     if isinstance(layer, torch.nn.Conv2d):
            #         layer.bias = layer.bias.to(torch.float32)

            pred = torch.sigmoid(output).cpu().numpy()
            predictedMask = (pred > 0.5).astype(np.float32)  # Apply threshold
            return pred.squeeze(), predictedMask.squeeze()  # Remove batch dimension

    def apply_mask_to_image(self, original_image, mask):
        """
        Applies the segmentation mask to the original image.
        The original image and the mask must have the same dimensions.
        """
        # Ensure mask is binary (0 and 1)
        binary_mask = (mask > 0).astype(np.uint8)

        # Apply mask to each channel of the original image
        if len(original_image.shape) == 3 and original_image.shape[2] == 3:
            segmented_image = np.zeros_like(original_image)
            for i in range(3):  # Assuming RGB image
                segmented_image[:, :, i] = original_image[:, :, i] * binary_mask
        else:
            segmented_image = original_image * binary_mask

        return segmented_image
    
    #pred = prediction from the model mask = predicted mask from the model
    def displaySegmentation(self, pred, mask):
        self.figure.clear()

        # Load the original image
        original_image = imageio.imread(self.current_image_path)

        # Load the ground truth mask if available
        ground_truth_mask = None
        if self.current_mask_path:
            ground_truth_mask = imageio.imread(self.current_mask_path)

        # Apply the masks to the original image
        original_segmented = self.apply_mask_to_image(original_image, ground_truth_mask) if ground_truth_mask is not None else None
        predicted_segmented = self.apply_mask_to_image(original_image, mask)

        # Create subplots for display
        ax1 = self.figure.add_subplot(2, 3, 1)  # Row 1, Column 1
        ax2 = self.figure.add_subplot(2, 3, 2)  # Row 1, Column 2
        ax3 = self.figure.add_subplot(2, 3, 3)  # Row 1, Column 3
        
        ax4 = self.figure.add_subplot(2, 3, 4)  # Row 2, Column 1
        ax5 = self.figure.add_subplot(2, 3, 5)  # Row 2, Column 2
        ax6 = self.figure.add_subplot(2, 3, 6)  # Row 2, Column 3

        # Display the original image and mask
        ax1.imshow(original_image, cmap='gray')
        ax1.axis('off')
        ax1.set_title("Original Image")

        if ground_truth_mask is not None:
            ax2.imshow(ground_truth_mask, cmap=self.get_random_colors(ground_truth_mask), interpolation='nearest')
            ax2.axis('off')
            ax2.set_title("Original Mask")
            ax3.imshow(original_segmented, cmap='gray')
            ax3.axis('off')
            ax3.set_title("Segmented with Original Mask")

        # Display the prediction and predicted mask
        ax4.imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax4.axis('off')
        ax4.set_title("Prediction")

        ax5.imshow(mask, cmap=self.get_random_colors(mask), interpolation='nearest')
        ax5.axis('off')
        ax5.set_title("Predicted Mask")

        ax6.imshow(predicted_segmented, cmap='gray')
        ax6.axis('off')
        ax6.set_title("Segmented with Predicted Mask")

        self.canvas.draw()
    
    def get_random_colors(self, labels):
        n_labels = len(np.unique(labels)) - 1
        cmap = [[0, 0, 0]] + np.random.rand(n_labels, 3).tolist()
        cmap = ListedColormap(cmap)
        return cmap

# def handle_exception(exc_type, exc_value, exc_traceback):
#     """
#     Handle uncaught exceptions and display them in a QMessageBox with 'Exit Application' and 'OK' buttons.
#     """
#     error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
#     msgBox = QMessageBox()
#     msgBox.setWindowTitle("Application Error")
#     msgBox.setText("An unexpected error occurred:\n" + error_msg)
#     msgBox.setStyleSheet("""
#         QMessageBox {
#             background-color: white;
#         }
#         QMessageBox QLabel {
#             background-color: white;
#         }
#         QMessageBox QPushButton {
#             background-color: white;
#         }
#     """)

#     # Create the 'Exit Application' button on the left
#     exit_button = msgBox.addButton("Exit Application", QMessageBox.RejectRole)

#     # Create the 'OK' button on the right
#     ok_button = msgBox.addButton(QMessageBox.Ok)

#     # Execute the message box and check the user's response
#     response = msgBox.exec_()

#     # Exit the application if the 'Exit Application' button was clicked
#     if msgBox.clickedButton() == exit_button:
#         QApplication.quit()
#     # msgBox.exec_()

class CustomMessageBox(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Error")

        # Create layout
        v_layout = QVBoxLayout(self)

        # Error message label
        self.error_label = QLabel("An unexpected error occurred:")
        v_layout.addWidget(self.error_label)

        # Button layout
        button_layout = QHBoxLayout()

        # Exit button
        self.exit_button = QPushButton("Exit Application")
        self.exit_button.clicked.connect(self.on_exit_clicked)
        button_layout.addWidget(self.exit_button, alignment=Qt.AlignLeft)

        # OK button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.on_ok_clicked)
        self.ok_button.setDefault(True)
        button_layout.addWidget(self.ok_button, alignment=Qt.AlignRight)

        v_layout.addLayout(button_layout)
        self.setLayout(v_layout)

    def set_error_message(self, message):
        self.error_label.setText(message)

    def on_exit_clicked(self):
        QApplication.quit()

    def on_ok_clicked(self):
        self.accept()

def handle_exception(exc_type, exc_value, exc_traceback):
    error_msg = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    msgBox = CustomMessageBox()
    msgBox.set_error_message("An unexpected error occurred:\n" + error_msg)
    msgBox.exec_()



    

def main():
    app = QApplication(sys.argv)

    # Override the exception hook with the custom function
    sys.excepthook = handle_exception

    ex = ImageSegmentationApp()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()