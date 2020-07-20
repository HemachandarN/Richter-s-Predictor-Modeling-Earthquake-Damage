####### Richter's Predictor: Modeling Earthquake Damage ##########

  
rm(list = ls()) # Clearing the global environment 

# Loding the Libraries
library(caret)
library(ggplot2)
library(corrplot)
library(rpart)
library(rpart.plot)
library(e1071)
library(randomForest)
library(fastAdaboost)
library(ipred)
library(DMwR)
library(ggplot2)
library(ggpubr)
library(nnet)

#Setting up the working Directoory 
setwd("D:/Quarter 3/Datamining/Project/Earthquake")


#Loading the data
data_ed<- read.csv("Earthquake_Damage.csv") 
data_ed

#Summary Statitics
str(data_ed)
summary(data_ed)


#Frequency of Damage_grade
table(data_ed$damage_grade)

ggplot(data_ed, aes(damage_grade)) +
  geom_bar(fill = "#FF6666") +
  theme_pubclean()
#There is a class Imbalance


## Data Preprocessing ##
 
#Check for Missing Values
nrow(data_ed[!complete.cases(data_ed),])
# No missing Values

## Outlier Detection and Removal ##

#Outliers
boxplot(data_ed$age,main = "Age")
boxplot(data_ed$area_percentage, main = "Area_percentage")
boxplot(data_ed$height_percentage, main = "Height_percentage")
boxplot(data_ed$count_floors_pre_eq, main = "Floor Count")
boxplot(data_ed$count_families, main = "Family Count")

# No of Rows Containing Outliers    
datau<-data_ed[abs(scale(data_ed$age)) >3,]
datav<- data_ed[abs(scale(data_ed$area_percentage)) >3,]
dataw <- data_ed[abs(scale(data_ed$height_percentage)) >3,]
datax<- data_ed[abs(scale(data_ed$count_floors_pre_eq)) >3,]
datay<-data_ed[abs(scale(data_ed$count_families)) >3,]

#Removing Outliers
data_a<-subset(data_ed, abs(scale(data_ed$age)) <= 3)
data_b <-subset(data_a, abs(scale(data_a$area_percentage)) <= 3)
data.c <- subset(data_a, abs(scale(data_a$height_percentage)) <= 3)
data.d <- subset(data_a, abs(scale(data_a$count_floors_pre_eq)) <= 3)
data <- subset(data_b,abs(scale(data_b$count_families)) <= 3)


##Categorical Variable to Dummy Variables##

#Below are the list of categorical variables
# land_surface_condition; 
# foundation_type 
# roof_type 
# ground_floor_type 
# other_floor_type 
# position  
# plan_configuration
# legal_ownership_status

dum<- dummyVars(~land_surface_condition+foundation_type+roof_type+ground_floor_type+other_floor_type+position+plan_configuration+legal_ownership_status,
                data = data, sep = "_", fullRank = TRUE)

df<- predict(dum,data)

data_dummy <- data.frame(data[,!names(data) %in% c('land_surface_condition','foundation_type','roof_type','ground_floor_type',
                                                   'other_floor_type','position','plan_configuration','legal_ownership_status')],df)

str(data_dummy)
cor(data_dummy)

## Removing Irrelevant Features ##

data_dummy$X<-NULL
data_dummy$building_id<-NULL

## Removing Redundant Features ##

colnames(data_dummy)
cor_vars <- cor(data_dummy[,-31])
findCorrelation(cor_vars, cutoff = 0.75, names = TRUE)

#Removing the redundant ones
data_dummy$count_floors_pre_eq<- NULL
data_dummy$position_s<-NULL
data_dummy$has_secondary_use<-NULL
data_dummy$plan_configuration_d<-NULL

## Balancing the dataset##
#There was a class imbalance in the dataset. Class 1 was less when compared with the other two classes

data_new <- data_dummy #Creating a copy

data_new$damage_grade <-as.factor(data_new$damage_grade) # Changing the dependent into a class variable

table(data_ed$damage_grade) #class distribution
barplot(table(data_new$damage_grade),col = heat.colors(3)) #Plot

#SMOTE
# We use the SMOTE() function from the DMwR package to perform
# Synthetic Minority Oversampling Technique

set.seed(17)
data_sm <- SMOTE(damage_grade~., data = data_new)

#Distribution after SMOTE
table(data_sm$damage_grade)
barplot(table(data_sm$damage_grade),col = heat.colors(3))


## Feature selection using Random Forest ##

set.seed(17)
data_final <- data_sm #Copy
str(data_final)



## Randowm Forest for feature selection ##

set.seed(17)
can.rf<- randomForest(damage_grade~.,data = data_final, importance = TRUE, proximity = TRUE, ntree = 500)
can.rf

can.rf$importance

varImpPlot(can.rf) # Plot showing variable Importance based on MeanDecreaseAccuracy and MeanDecreaseGini

# Based on the plot 8 variables were selected 
variable_selected <- data_final[,c("geo_level_1_id","geo_level_2_id","geo_level_3_id","age","area_percentage", "height_percentage","foundation_type_r","has_superstructure_mud_mortar_stone")]
names(variable_selected)

## New Dataset after feature selection 
data_final1 <- data_final[,colnames(data_final) %in% names(variable_selected)]

#Adding back the dependent variable 
data_final1$damage_grade <- data_final$damage_grade
names(data_final1)


## Standardising the Continuous variables ##

#Here we can see the data is in different scales for different variables 
#Hence we have to standardize all the variables to bring it to one scale.
#If we don't standardize the importance of one variable will be take over other variable in the analysis. 

standard<- data_final1[,c("geo_level_1_id", "geo_level_2_id", "geo_level_3_id", "age", "area_percentage","height_percentage")]
prepobj <- preProcess(x= standard, method = c('center', "scale"))
df1 <- predict(prepobj,standard)
summary(df1)
names(df1)

#Removing the unstandardized ones
data_final1$geo_level_1_id<-NULL
data_final1$geo_level_3_id<-NULL
data_final1$geo_level_2_id<-NULL
data_final1$age<-NULL
data_final1$area_percentage<-NULL
data_final1$height_percentage<- NULL

#Binding the standardized and the categorical ones reamaining in data_final1
data_mod <- cbind(df1,data_final1)
names(data_mod) #Final dataset for our modelling 
summary(data_mod)

################################################################################

#### Classification Models #####


### Decision Tree ###
  

## Splitting the dataset into Training and Testing

samp<- createDataPartition(data_mod$damage_grade, p = 0.8, list = FALSE)
train = data_mod[samp,]
test = data_mod[-samp,]

# Basic Classification Tree Model
set.seed(17)
cs.rpart<-rpart(formula = damage_grade~.,data = train, method = "class")
cs.rpart # Basic output of our tree

# Basic Tree Plot
plot(cs.rpart)
text(cs.rpart, cex=0.7)

# Alternate Version 
rpart.plot(cs.rpart, 
           extra = 2, 
           under = TRUE,  
           varlen=0, 
           faclen=0)


#Evaluating the Training Performance
tree.is.preds <- predict(object = cs.rpart, newdata = train, type = "class")
confusionMatrix(data= tree.is.preds,reference = train$damage_grade, mode = "prec_recall")

# We are getting an accuracy of 67%. We can further improve this model by model pruning. 

## Hyperparameter Tuning

# We want to tune the cost complexity parameter, or cp
# We choose the cp that is associated with the smallest
# cross-validated error (highest accuracy)

#Grid Search 

grids <- expand.grid(cp=seq(from=0,to=.25,by=.01))
grids

# We will use repeated 10-Fold cross validation and specify
# search="grid"
ctrl_grid <- trainControl(method="repeatedcv",
                          number = 10,
                          repeats = 3,
                          search="grid")


# Next, we create the DTFit object, which is our 
# cross-validated model, in which we are tuning the
# cp hyperparameter

set.seed(17)
DTFit <- train(form=damage_grade ~ ., 
               data = train, 
               method = "rpart",
               trControl = ctrl_grid, 
               tuneGrid=grids)

# We can view the basic summary for the DTFit object
DTFit

#Plot of cp value and Accuracy
plot(DTFit)

# variable importance information from our model
varImp(DTFit)

# Average confusion matrix across
confusionMatrix(DTFit) # Accuracy of 72%


# Random Search
# We will use repeated 10-Fold cross validation and specify
# search="random"
ctrl_random <- trainControl(method="repeatedcv",
                            number = 10,
                            repeats = 3,
                            search ="random")

# Train the model using random search
set.seed(17)
DT2Fit <- train(form = damage_grade ~ ., 
                data = train, 
                method = "rpart",
                trControl = ctrl_random, 
                tuneLength=10)

#Basic summary for the DT2Fit obect
DT2Fit

# Plot of cp value and Accuracy
plot(DT2Fit)

# Variable importance information from our model
varImp(DT2Fit)

#Average Confusion matrix across
# our resampled cross-validation models
confusionMatrix(DT2Fit) # Accuracy of 73%

#Random Search gave the best model
DT2Fit$finalModel

# We can plot the tree for our best fitting model. 
rpart.plot(DT2Fit$finalModel) # It will be tough to interpret

# Applying the best model to the training 
inpreds <- predict(object=DT2Fit, newdata=train)
confusionMatrix(data=inpreds, reference=train$damage_grade)

train_perf <- confusionMatrix(data=inpreds, 
                              reference=train$damage_grade, 
                              mode="prec_recall")
train_perf
# Accuracy of 79%
#Recall : Class1:0.8780;   Class2:0.7893;   Class3:0.6366;


#Applying the best model to our testing data
outpreds <- predict(object=DT2Fit, newdata=test)
confusionMatrix(data=outpreds, reference=test$damage_grade)

test_perf <- confusionMatrix(data=outpreds, 
                             reference=test$damage_grade,
                             mode="prec_recall")
test_perf
# Accuracy of 75%
# Recall: Class1:0.8619; Class2:0.7365;  Class3: 0.5506; 
##################################################################################################

## Artificial Neural Networks ##


# ANN performs well on standardised data. We have done that during the preprocessing stage. 


## Training & Testing
# Splitting the data into training and
# testing sets using a 80/20 split rule

set.seed(17)
samp2 <- createDataPartition(data_mod$damage_grade, p=.80, list=FALSE)
train_ann = data_mod[samp2, ] 
test_ann = data_mod[-samp2, ]


## Basic Model Building using the nnet Package

# We use the nnet() function from the nnet package

nnmod <- nnet(damage_grade~., data=train_ann, size=3, trace=FALSE)
names(train_ann)
ann.train <- predict(nnmod, 
                     train_ann[,-9],
                     type="class")

# Note: The predict() function here is returning a character vector
# which we need to convert to factor for our confusionMatrix()
confusionMatrix(factor(ann.train), 
                train_ann$damage_grade, 
                mode="prec_recall")

# We get an accuracy of 62%. We will try to improve this using hyperparameter tuning

## Hyperparameter Tuning
# We can use the caret package to tune our hyperparameters. 
# Here, we will use the nnet package. We can adjust the 
# size and decay.

# Size: number of nodes in the hidden layer. 
# (Note: There can only be one hidden layer using nnet)
# Decay: weight decay. regularization parameter to avoid overfitting, 
# which adds a penalty for complexity.

## Training the model using Repeated 5-fold Cross Validation and
# grid search

grids2 = expand.grid(size = seq(from = 1, to = 5, by = 1),
                    decay = seq(from = 0.1, to = 0.5, by = 0.1))

ctrl <- trainControl(method="repeatedcv",
                     number = 5,
                     repeats=3,
                     search="grid")

set.seed(17)
annMod <- train(damage_grade~ ., data = train_ann, 
                method = "nnet", 
                maxit=200,
                trControl = ctrl, 
                tuneGrid=grids2,
                verbose=FALSE)

annMod
plot(annMod)

confusionMatrix(annMod)
#Accuracy 65%

# We can apply our best fitting model to our training data
# to obtain predictions
inpreds <- predict(annMod, newdata=train_ann)

#Training Peformance
confusionMatrix(inpreds, train_ann$damage_grade, mode="prec_recall")

# Finally, we can apply our best fitting model to our
# testing data to obtain our outsample predictions
outpreds <- predict(annMod, newdata=test_ann)

#Testing Performance
confusionMatrix(outpreds, test_ann$damage_grade, mode="prec_recall")

# Overall the model performs badly for Class3 

####################################################################################################
save.image(file = "Final_Group6.RData")
load("Final_Group6.RData")
