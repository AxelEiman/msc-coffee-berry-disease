# Detecting coffee berry disease with machine learning -- Master thesis at RISE 2023 & 2024

The folder 'YOPO_PS' contains files for the Point-guided loss suppression model and 'YOPO_PT' contains the files for the Mixed point-teaching model. The files follow the same format as the YOLOv8 version **WE NEED TO ADD WHICH VERSION HERE AS SOME THINGS HAVE BEEN CHANGED AND PROBABLY WONT WORK WITH THE LATER VERSIONS** which this project was developed with. In each folder there are also files for using the respective model. The three most important files are:

*training.py* is the first place to get started, it is used to train a single model. The model and results from training are saved just as with a normal YOLO model. 

*loop_training.py* was used to loop through a directory containing subfolders with label files with different fractions of box and point labels and performs training with the data in each subfolder. Moves the images and and the yaml file to avoid having duplicate files.

*test.py* was used to test a set of models after training.
<!---
Vet inte om man borde ha med alla filer, tänker att man kan ge quick-start och sen får de lista ut resten själva? 
Kanske viktigare att beskriva typ mappstrukturen, eller vad typ yaml-filer osv innehåller om inte annat. Tycker vi skulle kunna ta bort det nedan iaf. Eller kanske bara skriva att det är bäst att utgå ifrån train.py om man vill förstå sig på hur allt går ihop?
--->


*train.py* is the main file during training. The loops throuch batches and epochs are done here. Easiest way to understand when the other files are called upon.

*loss.py* is were the new loss functions are initiated and also where the gain parameters for the MIL losses can be changed.

*tal.py* tasked aligned assigner is essientially where the predictions gets assigned to box or point labels.
