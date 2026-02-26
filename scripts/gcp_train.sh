# イメージ名の確認 (config.yaml に記載のもの)
IMAGE_URI="us-central1-docker.pkg.dev/r-nad-pokemontcgp/rnad-containers/r-nad-pokemontcgp:latest"

# ビルド
docker build -t $IMAGE_URI .

docker push $IMAGE_URI

gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=rnad-training-l4-spot \
  --config=config.yaml \
  --service-account=gce-819@r-nad-pokemontcgp.iam.gserviceaccount.com 