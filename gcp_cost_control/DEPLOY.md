# Deployment Instructions: GCP Cost Control for Vertex AI

This document provides the steps to deploy a Cloud Function that cancels all running/pending Vertex AI Custom Jobs when a billing alert is triggered.

## 1. Environment Variables

Set these in your terminal for easier execution:
```bash
PROJECT_ID="458093861942"
REGION="us-central1"
TOPIC_NAME="billing-alerts"
FUNCTION_NAME="cancel-vertex-jobs"
```

## 2. Create the Pub/Sub Topic

This topic will receive messages from the Cloud Billing Budget.

```bash
gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID
```

## 3. Deploy the Cloud Function (Gen 2)

Run this command from within the `gcp_cost_control` directory:

```bash
gcloud functions deploy $FUNCTION_NAME \
    --gen2 \
    --region=$REGION \
    --runtime=python310 \
    --trigger-topic=$TOPIC_NAME \
    --entry-point=cancel_vertex_jobs \
    --project=$PROJECT_ID \
    --source=.
```

## 4. Grant IAM Permissions

The Cloud Function needs permission to list and cancel Vertex AI jobs. By default, Gen 2 functions use the Compute Engine default service account.

First, determine the service account name. It is usually:
`458093861942-compute@developer.gserviceaccount.com`

Grant the **Vertex AI Administrator** role (or a more restricted role that includes `aiplatform.customJobs.cancel` and `aiplatform.customJobs.list`):

```bash
SERVICE_ACCOUNT="458093861942-compute@developer.gserviceaccount.com"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/aiplatform.admin"
```

## 5. Configure Billing Alert (Manual Step)

1. Go to the [Google Cloud Console Billing section](https://console.cloud.google.com/billing).
2. Navigate to **Budgets & alerts**.
3. Edit your budget or create a new one.
4. Under **Actions**, check "Connect a Pub/Sub topic to this budget".
5. Select the topic you created: `billing-alerts`.
6. Save the budget.

Now, whenever the budget threshold is met, the Pub/Sub message will trigger the Cloud Function and cancel your expensive Vertex AI jobs automatically.
