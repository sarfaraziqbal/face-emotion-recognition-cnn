# Introduction

We have built a deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

# Dependencies

Python 3, OpenCV, Tensorflow, 
streamlit-webrtc

# Explanation

The dataset which I used was the FER2013 dataset from kaggle. You can download the dataset from the link below and copy paste the dataset in the folder. https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset


![name-of-you-image](https://raw.githubusercontent.com/shreyasah99/Shreyas-Deep-Learning-Capstone-Face-Emotion-Recognition/main/Structure%20of%20model.png)

This was the model structure. In the output layer there were 7 nodes. This model was used to predict emotion in following ways:

1. First, the haar cascade method is used to detect faces in each frame of the webcam feed.
2. The region of image containing the face is resized to 48x48 and is passed as input to the CNN.
3. The network outputs a list of softmax scores for the seven classes of emotions.
4. The emotion with maximum score is displayed on the screen.
4. This model gave a training accuracy of 66.47 and validation accuracy of 58.19 after 42 epocs.

The model which my friend Apoorva trained was with the help of transfer learning and she got training accuracy of 79.42 with validation accuracy of 75.94 at just 30 epocs, but was having an issue of memory i.e. slug size and got application error during deployment but was deployed succesfully. You can access the code for her model from the below github link: https://github.com/Apoorva2399/Real-Time-Face-Emotion-Recogniton


Then we made frontend of the model on streamlit and then deployed the model on heroku cloud as well.

Link of repository containing streamlit code: https://github.com/sarfaraziqbal/face-emotion-recognition-cnn/

# Deployment
In this repository we have made a front end using streamlit. Streamlit doesn???t provide the live capture feature itself, instead uses a third party API. We used streamlit-webrtc which helped to deal with real-time video streams. Image captured from the webcam is sent to VideoTransformer function to detect the emotion.

Then this model was deployed on heroku platform with the help of buildpack-apt which is necessary to deploy opencv model on heroku. But heroku platform only allows model size as 500 mb. And tensorflow 2.0 itself takes 420 mb so we replaced it with tensorflow-cpu. All the other packages used and their version can be found in requirements.txt . Our final model was of 414 mb and it was successfully deployed but the live stream itself takes max. of 300 mb while loading live-stream or opening the webcam. And hence the webcam is loading but not capturing the faces so, our model was not giving expected output.

Link of our model that's deployed on heroku: https://model37.herokuapp.com/
