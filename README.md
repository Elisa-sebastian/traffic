# hands-on-2021
**The aim of the project is to build 2 different image classifiers in order to recognize traffic signs.**

*Comparing two models we ended up with:* 
* Accuracy of **95.57%** on the test set for **DNN** 
* Accuracy of **87.57%** on the test set for **SVM** 

**DNN has difficulties to recognize:**
- Traffic sign ***'End of no passing'*** *(class 41)*. It is confused with traffic sign ***'No passing'*** *(class 9).* 
- Sometimes traffic sign ***'End of speed limit (80km/h)'*** *(class 6)* is confused with traffic sign ***'End no passing veh > 3.5 tons'*** *(class 42)*

**As for SVM,** it has a lot more difficulties to recognize correctly traffic signs, but **it performs better to recognize the traffic sign *'End of no passing'*** *(class 41)* where DNN is less performant. 

## Install

* Clone this repository
* Download images with 'scripts/download_images.sh'
* Create environment with needed modules. All used modules in **requirements.txt**

## Repertory parameter
* parameters.yaml: configuration yaml file a file search pattern and some parameters for classifiers

## Repertory notebooks
* Deep Neural Network and in interface to classify traffic signs *(train_model_DNN)*
* Support Vector Machine classifier *(train_model_SVM)*

## Repertory Models
To save time, we propose  our already **trained and saved models:** 

* DNN : traffic_signs_2021-03-20_09-40-32.h5
* SVM : traffic_signs_svm.pickle 

## Repertory app
We propose an application that allows the user to upload their image and see how the traffic sign is classified by the DNN. 

* app.py *(with app.yaml for parameters and access to the data)* 

## References

* Dataset introduction: https://benchmark.ini.rub.de/gtsrb_dataset.htm
* Images: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html
