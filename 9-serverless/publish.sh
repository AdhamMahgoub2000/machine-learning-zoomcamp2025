
ECR_URL=436667402240.dkr.ecr.us-east-1.amazonaws.com
REMOTE_IMAGE_TAG=${ECR_URL}/hair_style_classifier:v1

aws ecr get-login-password \
  --region us-east-1 \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

docker build -t hair_style_classifier .
docker tag hair_style_classifier ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "done"