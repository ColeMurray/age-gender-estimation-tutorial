# age-gender-estimation-tutorial
Tutorial for creating multi-task age and gender estimator in Tensorflow
https://medium.freecodecamp.org/how-to-build-an-age-and-gender-multi-task-predictor-with-deep-learning-in-tensorflow-20c28a1bd447


### Pulling the image
docker pull -t colemurray/age-gender-estimation-tutorial 

### Building the image
```
# CPU:
docker build -t colemurray/age-gender-estimation-tutorial .

# GPU
docker build -t colemurray/age-gender-estimation-tutorial -f Dockerfile.gpu .
```
##### 
