# rock-paper-scissors-but-with-your-actual-hand
Keras + OpenCV implementation of the iconic rock paper scissors game using a convolutional neural network and optical flow to play using your actual hand (and a camera). The machine to play against is extracted from http://www.rpscontest.com/

Hand gesture recognition based on: https://github.com/anantSinghCross/realtime-hand-gesture-recognition

Demo:

[![Alt text](https://img.youtube.com/vi/vl_M1duWr8k/0.jpg)](https://www.youtube.com/watch?v=vl_M1duWr8k&)


## FAQ

**Q: How does it work?**

A: There are four steps to this project (which individually are in no way innovative or new). First, the segmentation of the hand using OpenCV. Second, predicting the gesture using a convolutional neuronal network (while the CNN is most likely neccesary, the number of nodes of the convolutional layer is probably an overkill). Third, to determine the movement of the hand and recognize the swinging motion using optical flow (again, an overkill, but I just pasted my code from another assignment). Lastly, pass the chosen gesture to the bot and let the magic flow.

**Q: How do I run it myself?**

A: Supposing you have all of the packages installed: 

python rps_cnn.py rps_model.json rps_model_weights.h5 (and optionally, the camera source using --dev=(source))

**Q: What is the accuracy of the model?**

A: 100% on the second epoch using 1000 images per gesture and a 90-10 train-test ratio. I thought there was overfitting but apparently there is not. I also forgot to take a full screenshot: https://i.imgur.com/m3DQBnm.png

Keep in note the dataset is very poor as it only consists of my right hand.

**Q: Could you provide us the dataset?**

A: I accidentally deleted it and cannot be bothered to remake it again. You can do it by yourself using `rps_generate_dataset.py`. It should look something like this:
https://i.imgur.com/r3R7oBh.png


**Q: What are the requirements and dependencies?**

A: Keras/TensorFlow, OpenCV, numpy, umucv

I do not know the proper nomenclature to state dependencies and my packages are a mess due to other projects. Installing Anaconda and then using pip install to add any missing packages should do the trick.

**Q: Why is X done like Y when it could be done like Z, which is way more efficient and straightforward?**

A: This is a university assignment and I am not particularly proficient in ~~parsel~~ Python.

**Q: What is this umucv package?**

A: It is a simple utility developed by Alberto Ruiz (http://dis.um.es/profesores/alberto/) that simplifies some of the steps (capturing the camera device, selecting regions of interest, etc)

the package itself: pip install --upgrade http://robot.inf.um.es/material/umucv.tar.gz


