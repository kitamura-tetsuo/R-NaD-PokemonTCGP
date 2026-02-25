gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=rnad-training-tpu-spot \
  --config=config.yaml \
  --service-account=gce-819@r-nad-pokemontcgp.iam.gserviceaccount.com 