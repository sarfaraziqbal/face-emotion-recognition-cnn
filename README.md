# Introduction

We have built a deep learning model which detects the real time emotions of students through a webcam so that teachers can understand if students are able to grasp the topic according to students' expressions or emotions and then deploy the model. The model is trained on the FER-2013 dataset .This dataset consists of 35887 grayscale, 48x48 sized face images with seven emotions - angry, disgusted, fearful, happy, neutral, sad and surprised.

#Dependencies

Python 3, OpenCV, Tensorflow

#Model Explanation

The dataset which I used was the FER2013 dataset from kaggle. You can download the dataset from the link below and copy paste the dataset in the folder. https://www.kaggle.com/jonathanoheix/face-expression-recognition-dataset



This was the model structure. In the output layer there were 7 nodes. This model was used to predict emotion in following ways:

1. First, the haar cascade method is used to detect faces in each frame of the webcam feed.
2. The region of image containing the face is resized to 48x48 and is passed as input to the CNN.
3. The network outputs a list of softmax scores for the seven classes of emotions.
4. The emotion with maximum score is displayed on the screen.
4. This model gave a training accuracy of 60.3 and testing accuracy of 57.7.

The model which my friend Apoorva made was with the help of transfer learning and he got an accuracy of 70, hence we decided that further projects i.e. frontend and deployment will be done on her model. You can access the code for her model from the below github link: https://github.com/jatin090/Jatin-Deep-Learning-Capstone-Face-Emotion-Recognition.git

Then we made frontend of the model on streamlit these models on heroku cloud as well.

Link of repository containing streamlit code: https://github.com/shreyasah99/facial-emotion-detect-with-streamlit.git
