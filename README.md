# Installation:
1.pip install -r requirements.txt

# How to run:
## Option 1 (without retraining the models):
1.Run the FinalGui.py<br/>
2.In the login option use the following:
 - user:tal<br/>password:123<br/>
 [Image 1](https://drive.google.com/file/d/1Uxg2L6cKRvfYnW_2JBG67OqSyAWdyXHY/view?usp=sharing)
 
## Option 2 (with retraining the models):
# ''' - this will take at least one hour to retrain bouth models:'''
1.Run the FinalGui.py<br/>
2.Click on register and followe the program instruction<br/>
[Image 1](https://drive.google.com/file/d/1Uxg2L6cKRvfYnW_2JBG67OqSyAWdyXHY/view?usp=sharing)<br/>
[Image 2](https://drive.google.com/file/d/1vNUS3BmP6CFCvAgU9Qxn6r6FYn0M1PZL/view?usp=sharing)<br/>
[Image 3](https://drive.google.com/file/d/1KLT04H2RmdIYnB0asFdkiYsfNz2apS1e/view?usp=sharing)<br/>

# DATASET:
## Word recognition data:
**Source**:<br/>
The data is a combination of recordings that has been generated by text to speech program in different accents and manual recordings.<br/>
**Labels**: <br/>
In total there are 41 labels, 40 of them are people's names and the last one is a noise label which helps us prevent false predictions.<br/>
**Description**:<br/>
The samples are in wav format in average length of 1 second.<br/>
There are 13 women samples and 4 men samples for each label.<br/>
**Number of samples**:<br/>
The number of the original samples is 680, for each label we expend the data and add 391 samples thus each label has 408 samples.<br/>
In addition we added a noise label that has 2000 samples. <br/>
The total amount of samples is 18320.<br/>
**Dataset Location**:<br/>
Due to large size of the data,we created a zip file it's located in this [google drive link](https://meet.google.com/linkredirect?authuser=0&dest=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1nNRhXT5ko_dLm2eyBCIyA6FC7Je5SF28%3Fusp%3Dsharing)</br>
If you want to retrain the modles you need to download the zip,and extract the folder into main folder project.

## Speaker recognition data:
**Source**:<br/>
The data is a collection of manual recordings made by 2 women and 2 men and TIMID database with mixed genders.<br/>
**Labels**: <br/>
In total there are 5 labels, 4 of them are people's names and the last one is an unidentified label for people that are not in the system.<br/>
Description: The samples are in PNG format with an average size of 10.5 KB.<br/>
**Number of samples**:<br/>
For each label there are 15 samples, in total there are 75 samples.<br/>
**Dataset Location**:<br/>
The data is located in this [link](https://github.com/thewolfe1/TheThirdEye/tree/master/speaker)<br/>

# Project book:
[link](https://github.com/thewolfe1/TheThirdEye/blob/master/%D7%A1%D7%A4%D7%A8%20%D7%A4%D7%A8%D7%95%D7%99%D7%99%D7%A7%D7%98.pdf)</br>

# Project demo video:
[link](https://github.com/thewolfe1/TheThirdEye/blob/master/TheThirdEye.mp4)</br>
