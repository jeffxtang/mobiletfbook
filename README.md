# About
Source code repo for the book [Intelligent Mobile Projects with TensorFlow](https://www.amazon.com/Intelligent-Mobile-Projects-TensorFlow-wide-ranging/dp/1788834542).

The iOS apps have been tested with Xcode 8.2.1 and 9.2, and the Android apps have been tested with Android Studio 3.0.1. TensorFlow 1.4, 1.5, 1.6, and 1.7 have been used for testing.

# How to run the apps

1. To get the repo, do `git clone https://github.com/jeffxtang/mobiletfbook`. The whole repo takes about 186MB, not including the large trained TensorFlow model files used in the iOS and Android apps in Chapters 3, 6, 9, and 11 - you can download the large model files in a 1.12GB zip format from my Google Drive [here](https://drive.google.com/file/d/1ARnO_Dhhkzhia5SA4gn0mEIHCFCEN-tJ). After downloading and unzipping it (to a folder named large_files), drag and drop all the folders inside the large_files folder to the mobiletfbook folder, created when you run `git clone https://github.com/jeffxtang/mobiletfbook`, to merge the large models into their relevant locations of the apps using them.

2. To run the iOS apps in Chapters 2 and 6, which use the TensorFlow pod, first open a Terminal and cd to the app's project folder where the Podfile is located, then run `pod install`. After that, open the .xcworkspace file in Xcode.

3. To run the iOS apps in Chapters 3, 4, 5, 7, 8, 9, and 10, open the app's .xcodeproj file in Xcode, then go to the app's PROJECT's Build Settings, make sure the TENSORFLOW_ROOT variable is set to your TensorFlow source root. By default, it's set to $HOME/tensorflow-1.7.0 for all iOS apps in Chapters 4, 5, and 7-10, and to $HOME/tensorflow-1.4.0 for the iOS app in Chapter 3 (to be consistent with the protobuf version used for the app).

4. To run the Android apps in Chapters 2-11 (except Chapter 3 where only an iOS app is included), simply select "Open an existing Android Studio project" after launching Android Studio, then choose the Chapter's android folder. If you see an error "Error Loading Project: Cannot load 2 modules" with details as "2 modules cannot be loaded. You can remove them from the project (no files will be deleted).", you can ignore it or click Remove Selected.

5. To run the TensorFlow Lite, Core ML and Raspberry Pi apps in Chapters 11 and 12, please follow the steps in the book.

6. If you still use Xcode 8.2.1, probably because you want to see if your older Mac with an older OS version such as OS X 10.11 El Capitan can run the latest and greatest TensorFlow, then you may encounter an error "Base.lproj/Main.storyboard: This document requires Xcode 9.0 or later." when building an iOS app in the repo. To fix it and let your older Mac shine with the power of TensorFlow, simply drag the two stobyboard files inside the repo's Xcode821 folder to the iOS app's Base.lproj folder and rebuild the app. Warning: building the TensorFlow iOS custom library, as required by many iOS apps in the book, could take 2-3 hours on an older Mac, but after that, running the iOS apps works like a breeze.

# What's included in the book
There are 12 chapters in the book, starting with steps on how to set up TensorFlow and a NVIDIA GPU for much faster TensorFlow model training, as well as Xcode and Android Studio for TensorFlow Mobile app development. Chapters 2 to 10 are about building and training TensorFlow models and use the models on iOS and Android apps using TensorFlow Mobile, the current production-ready way to run TensorFlow apps on mobile (versus TensorFlow Lite, an alternative way - see the section When to Read the Book for more details).

First, a quick summary of Chapters 2 to 12 for the impatient.

In Chapters 2, 3 and 4,  you'll see step-by-step tutorials of how to retrain and use TensorFlow models, included in the TensorFlow example apps, in your own iOS and Android apps. Both Objective-C and Swift based iOS apps, using both the TensorFlow pod and the manually built TensorFlow library, are provided.

Chapters 5, 6, and 7 offer detailed tutorials of using 3 additional TensorFlow models, implemented in Python and described in the official TensorFlow tutorials, in your iOS and Android apps.

Chapter 8 introduces a recurrent neural network (RNN) model built from scratch in TensorFlow and Keras for stock price prediction. Chapter 9 shows you how to build and train GAN models, one of the most exciting advances in deep learning in recent years, and use them in your mobile apps. Chapter 10 covers how to build and train an AlphaZero-like model in iOS and Android mobile game apps (AlphaZero is the latest and best model using deep reinforcement learning based on AlphaGo).

In Chapter 11, you'll see how to use TensorFlow Lite, an alternative solution, but still in developer preview, to TensorFlow Mobile, in iOS and Android apps. Core ML for iOS is also covered.  In Chapter 12, the final chapter of the book, you'll learn how to set up Raspberry Pi and GoPiGo, and how to install TensorFlow and run TensorFlow models on Raspberry Pi to make it move, see, listen and speak. You'll also see in-depth discussion of reinforcement learning using policy gradient, with model built with TensorFlow and running on Raspberry Pi.

Detailed topics for each chapter are listed as follows.

Chapter 1, Getting Started with Mobile TensorFlow. We'll discuss how to set up TensorFlow on Mac and Ubuntu and NVIDIA GPU on Ubuntu and how to set up Xcode and Android Studio. We'll also discuss the difference between TensorFlow Mobile and TensorFlow Lite and when you should use which. Finally, we'll show you how to run the sample TensorFlow iOS and Android apps.

Chapter 2, Classifying Images with Transfer Learning. We'll discuss what is transfer learning and why you should use it, how to retrain the Inception v3 and MobileNet models for more accurate and faster dog breed recognition, and how to use the retrained models in sample iOS and Android apps. Then we'll show you how to add TensorFlow to your own iOS app, both in Objective-C and Swift, and as well as your own Android app for dog breed recognition.

Chapter 3, Detecting Objects and Their Locations. We'll first give a quick overview of Object Detection, then show you how to set up the TensorFlow Object Detection API and use it to retrain SSD-MobileNet and Faster RCNN models. We'll also show you how to use the models, which are used in the example TensorFlow Android app, in your iOS app by manually building the TensorFlow iOS library to support non-default TensorFlow operations. Finally, we'll show you how to train YOLO2, another popular object detection model, which is also used in the example TensorFlow Android app, and how to use it in your iOS app.

Chapter 4, Transforming Pictures with Amazing Art Styles. We'll first give an overview of neural style transfer with their rapid progress in the last few years, then show you how to train fast neural style transfer models and use them in iOS and Android apps. After that, we'll cover how to use the TensorFlow Magenta multi-style model in your own iOS and Android apps to easily create amazing art styles.

Chapter 5, Understanding Simple Speech Commands. We'll give a quick overview of speech recognition, and show you how to train a simple speech commands recognition model. We'll then show you how to use the model in Android, as well as in iOS using both Objective-C and Swift. We'll also cover more tips on how to fix possible model loading and running errors on mobile.

Chapter 6, Describing Images in Natural Language. We'll first describe how image captioning works, then show you how to train and freeze an image captioning model in TensorFlow. We'll further discuss how to transform and optimize the complicated model to get it ready for running on mobile. Finally, we'll offer complete iOS and Android apps using the model to generate natural language description of images.

Chapter 7, Recognizing Drawing with CNN and LSTM. We'll first cover how drawing classification works, and discuss how to train, predict and prepare the model. Then we'll show you how to build another custom TensorFlow iOS library to use the model in a fun iOS doodling app. Finally, we'll show you how to build the custom TensorFlow Android library to fix a new model loading error and then use the model in your own Android app.

Chapter 8, Predicting Stock Price with RNN. We’ll first discuss RNN and how to use it to predict stock prices, then we'll show you how to build a RNN model with the TensorFlow API to predict stock prices, and how to build a RNN LSTM model with the easier-to-use Keras API to achieve the same goal. We'll test and see if such models can beat a random buy or sell strategy. Finally, we'll show you how to run the TensorFlow and Keras models in both iOS and Android apps.

Chapter 9, Generating and Enhancing Images with GAN. We'll first give an overview of what GAN is and why it has such great potential. Then we'll show you how to build and train a basic GAN model that can be used to generate human-like handwritten digits and a more advanced model that can enhance low resolution images to high resolution ones. We'll finally cover how to use the two GAN models in your iOS and Android apps.

Chapter 10, Building AlphaZero-like Mobile Game App. We'll first discuss how the latest and coolest AlphaZero works, and how to train and test a AlphaZero-like model to play a simple but fun game called Connect 4 in Keras with TensorFlow as backend. We'll then show you the complete iOS and Android apps to use the model and play the game Connect 4 on your mobile devices.

Chapter 11, Using TensorFlow Lite and Core ML on Mobile. We'll first give an overview of TensorFlow Lite, then show you how to use a pre-built TensorFlow model, a retrained TensorFlow model for TensorFlow Lite, and a custom TensorFlow Lite model in iOS. We'll also show you how to use TensorFlow Lite in Android. After that, we'll give an overview of Apple's Core ML, and show you how to use Core ML with standard machine learning using Scikit-Learn. Finally, we'll cover how to use Core ML with TensorFlow and Keras.

Chapter 12, Developing TensorFlow Apps on Raspberry Pi. We'll first show you how to set up Raspberry Pi and make it move, and how to set up TensorFlow on Raspberry Pi. Then we'll cover how to use the TensorFlow image recognition and audio recognition models, along with text to speech and robot movement APIs, to build a Raspberry Pi robot that can move, see, listen, and speak. Finally, we'll discuss in detail how to use OpenAI Gym and TensorFlow to build and train a powerful neural network based reinforcement learning policy model from scratch in a simulated environment to make the robot learn to keep its balance.

# Questions or comments
Please create an issue or contact me at jeff@ailabby.com or jeff.x.tang@gmail.com. Your feedback is always welcome and appreciated!
