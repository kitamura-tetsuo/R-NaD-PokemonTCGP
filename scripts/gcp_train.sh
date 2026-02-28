# イメージ名の確認 (config.yaml に記載のもの)
IMAGE_URI="us-central1-docker.pkg.dev/r-nad-pokemontcgp/rnad-containers/r-nad-pokemontcgp:latest"

# ビルド
docker build -t $IMAGE_URI .
# docker build --no-cache -t $IMAGE_URI .

docker push $IMAGE_URI

# 古い（タグのない）イメージを削除
# IMAGE_BASE=${IMAGE_URI%:*}
# gcloud artifacts docker images list "$IMAGE_BASE" --filter='-tags:*' --format='value(version)' | xargs -r -I {} gcloud artifacts docker images delete "$IMAGE_BASE@{}" --quiet

JOB_ID=$(gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=rnad-training-l4-spot \
  --config=config.yaml \
  --service-account=gce-819@r-nad-pokemontcgp.iam.gserviceaccount.com \
  --format="value(name)")

if [ -n "$JOB_ID" ]; then
  echo "Job created successfully: $JOB_ID"
  gcloud ai custom-jobs stream-logs "$JOB_ID" --region=us-central1
else
  echo "Failed to create job."
  exit 1
fi