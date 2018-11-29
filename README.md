# age-gender-estimation-tutorial
Tutorial for creating multi-task age and gender estimator in Tensorflow


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
