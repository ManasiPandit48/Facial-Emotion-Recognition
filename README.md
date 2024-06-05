# Facial-Emotion-Recognition-

This project involves creating and training a Convolutional Neural Network (CNN) to detect emotions from facial expressions using the FER 2013 dataset. The trained model can then be used to perform real-time emotion detection from a webcam feed.

## Requirements
The following packages are required to run this project:
- Python 3.7+
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pillow

You can install the required packages using pip:
```bash
pip install numpy opencv-python keras tensorflow pillow
```
### Download FER2013 Dataset

Download the FER 2013 dataset from Kaggle using the following link and place it in the `data` folder under your project directory:
[Download FER2013 Dataset](https://www.kaggle.com/msambare/fer2013)

## Training the Emotion Detection Model

The `TrainEmotionDetector.py` script is used to train the emotion detection model. It reads the FER 2013 dataset, builds a Convolutional Neural Network (CNN) model, trains the model on the training data, and saves the trained model architecture and weights.

### Training Steps:

1. **Initialize Data Generators**: Create data generators for training and validation data with rescaling.
2. **Build the CNN Model**: Define a sequential model with convolutional layers, max-pooling layers, dropout layers, and dense layers.
3. **Train the Model**: Fit the model using the training data and validate using the validation data.
4. **Save the Trained Model**: Save the model architecture as a JSON file and the model weights as an HDF5 file.

### Running the Training Script:

To train the model, execute the following command:

```bash
python TrainEmotionDetector.py
```

The training process may take several hours depending on your hardware. For instance, on an Intel i7 processor with 16 GB of RAM, it might take around 4 hours.

After training, the trained model structure and weights will be saved in your project directory:
- `emotion_model.json`
- `emotion_model.h5`

Move these files into the `Model` directory in your project.

## Testing the Emotion Detection Model

The `TestEmotionDetector.py` script is used to test the trained model. It loads the model architecture and weights, captures real-time video from the webcam, detects faces, and predicts emotions.

### Testing Steps:

1. **Load the Trained Model**: Load the model architecture and weights from the saved JSON and HDF5 files.
2. **Start the Webcam Feed**: Capture real-time video from the webcam.
3. **Detect Faces**: Use OpenCV's Haar Cascade Classifier to detect faces in the video feed.
4. **Predict Emotions**: For each detected face, predict the emotion using the trained model.
5. **Display the Results**: Draw rectangles around faces and display the predicted emotions on the video feed.

### Running the Testing Script:

To run the emotion detection in real-time, execute the following command:

```bash
python TestEmotionDetector.py
```

Press the 'q' key to exit the real-time emotion detection.

## Notes

- Ensure the FER 2013 dataset is placed correctly in the `data/train` and `data/test` directories with appropriate subfolders for each emotion.
- The model and weights should be located in the `Model` directory after training.
- Make sure your webcam is properly connected and accessible for real-time testing.

## Acknowledgments

- The FER 2013 dataset used in this project is available on [Kaggle](https://www.kaggle.com/msambare/fer2013).
- This project uses Keras for building and training the neural network and OpenCV for real-time emotion detection from webcam feed.
