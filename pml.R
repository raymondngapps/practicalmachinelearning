# import library
library(caret)

set.seed(1000)

# Read data from data file and remove obviously error in data and set it to NA
pml_raw <- read.csv("pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
pml_forecast <- read.csv("pml-testing.csv", na.strings = c("NA", "#DIV/0!"))

# It seems not make sense to use the user_name as one of the variable to predict the outcome
pml_raw <- subset(pml_raw, select = -user_name)
pml_forecast <- subset(pml_forecast, select = -user_name)

# Calculate the ratio of NA for a particular and remove those mostly na
print("Sanitise data")

naCols <- colSums(data.matrix(!is.na(pml_raw))) / nrow(pml_raw)
pml_raw <- pml_raw[naCols > 0.1]
pml_forecast <- pml_forecast[naCols > 0.1]

# Net result is like this
# 
# X raw_timestamp_part_1 raw_timestamp_part_2       cvtd_timestamp           new_window           num_window 
# 1                    1                    1                    1                    1                    1 
# roll_belt           pitch_belt             yaw_belt     total_accel_belt         gyros_belt_x         gyros_belt_y 
# 1                    1                    1                    1                    1                    1 
# gyros_belt_z         accel_belt_x         accel_belt_y         accel_belt_z        magnet_belt_x        magnet_belt_y 
# 1                    1                    1                    1                    1                    1 
# magnet_belt_z             roll_arm            pitch_arm              yaw_arm      total_accel_arm          gyros_arm_x 
# 1                    1                    1                    1                    1                    1 
# gyros_arm_y          gyros_arm_z          accel_arm_x          accel_arm_y          accel_arm_z         magnet_arm_x 
# 1                    1                    1                    1                    1                    1 
# magnet_arm_y         magnet_arm_z        roll_dumbbell       pitch_dumbbell         yaw_dumbbell total_accel_dumbbell 
# 1                    1                    1                    1                    1                    1 
# gyros_dumbbell_x     gyros_dumbbell_y     gyros_dumbbell_z     accel_dumbbell_x     accel_dumbbell_y     accel_dumbbell_z 
# 1                    1                    1                    1                    1                    1 
# magnet_dumbbell_x    magnet_dumbbell_y    magnet_dumbbell_z         roll_forearm        pitch_forearm          yaw_forearm 
# 1                    1                    1                    1                    1                    1 
# total_accel_forearm      gyros_forearm_x      gyros_forearm_y      gyros_forearm_z      accel_forearm_x      accel_forearm_y 
# 1                    1                    1                    1                    1                    1 
# accel_forearm_z     magnet_forearm_x     magnet_forearm_y     magnet_forearm_z               classe 
# 1                    1                    1                    1                    1 

# Split train set and test set
inTrain <- createDataPartition(y = pml_raw$classe, p = 0.7)[[1]]
pml_train <- pml_raw[inTrain, ]
pml_test <- pml_raw[-inTrain, ]

# Random forest do not handle missing data
print("Create model")
modForest <- train(classe ~ ., data=pml_train, method = "rf")

# The model takes very long
print("Predict model")
predictForest <- predict(modForest, newdata=pml_test)

print("Verify model")
print(confusionMatrix(as.factor(pml_test$classe), as.factor(predictForest)))


# Confusion Matrix and Statistics

# Reference
# Prediction    A    B    C    D    E
# A 5014    8    0    0    0
# B    0 3417    0    0    0
# C    0    3 3076    0    0
# D    0    0    0 2894    0
# E    0    0    0    0 3246

# Overall Statistics

# Accuracy : 0.9994          
# 95% CI : (0.9989, 0.9997)
# No Information Rate : 0.284           
# P-Value [Acc > NIR] : < 2.2e-16       

# Kappa : 0.9992          

# Mcnemar's Test P-Value : NA              

# Statistics by Class:

#                      Class: A Class: B Class: C Class: D Class: E
# Sensitivity            1.0000   0.9968   1.0000   1.0000   1.0000
# Specificity            0.9994   1.0000   0.9998   1.0000   1.0000
# Pos Pred Value         0.9984   1.0000   0.9990   1.0000   1.0000
# Neg Pred Value         1.0000   0.9992   1.0000   1.0000   1.0000
# Prevalence             0.2840   0.1941   0.1742   0.1639   0.1838
# Detection Rate         0.2840   0.1935   0.1742   0.1639   0.1838
# Detection Prevalence   0.2844   0.1935   0.1744   0.1639   0.1838
# Balanced Accuracy      0.9997   0.9984   0.9999   1.0000   1.0000
