# Deep learning based heart rate estimation from PPG data with motion artifacts

Summary: Fitness/health trackers use optical sensors called PPG sensors to measure the heart rate and other health metrics of the user. For a smartwatch, the PPG sensor is located on the back making contact with the user's skin. When resting or sleeping, the sensor-skin contact is static resulting in accurate heart rate measurements. During fitness activities the sensor-skin contact changes adding what's called motion artifacts to the PPG sensor measurements. This repo implements an end-end deep learning method to estimate the heart rate accurately during dynamic scenarios. The architecture used here is an implementation of the following paper: 

- Dataset: Troika
- Paper: 

References:
- https://github.com/KJStrand/Pulse_Rate_Estimation