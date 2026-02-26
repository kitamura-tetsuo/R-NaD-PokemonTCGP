import base64
import functions_framework
from google.cloud import aiplatform
from cloudevents.http import CloudEvent

# Configuration
PROJECT_ID = "458093861942"
REGION = "us-central1"

@functions_framework.cloud_event
def cancel_vertex_jobs(cloud_event: CloudEvent):
    """
    Cloud Function triggered by Pub/Sub to cancel running/pending Vertex AI Custom Jobs.
    """
    print(f"Received CloudEvent ID: {cloud_event['id']}")
    
    # Initialize Vertex AI SDK
    aiplatform.init(project=PROJECT_ID, location=REGION)

    try:
        # Define the states to filter for
        # States: 1 (PENDING), 2 (QUEUED), 3 (RUNNING), etc.
        # Filter syntax documentation: https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.customJobs/list
        filter_str = 'state="JOB_STATE_RUNNING" OR state="JOB_STATE_PENDING"'
        
        print(f"Listing Vertex AI Custom Jobs in {REGION} with filter: {filter_str}")
        custom_jobs = aiplatform.CustomJob.list(filter=filter_str)

        if not custom_jobs:
            print("No running or pending Custom Jobs found.")
            return

        print(f"Found {len(custom_jobs)} job(s) to cancel.")

        for job in custom_jobs:
            try:
                print(f"Cancelling job: {job.display_name} (ID: {job.resource_name})")
                job.cancel()
                print(f"Cancellation initiated for job: {job.resource_name}")
            except Exception as e:
                print(f"Failed to cancel job {job.resource_name}: {str(e)}")

    except Exception as e:
        print(f"An error occurred during job processing: {str(e)}")
        # We don't want to crash the function entirely if one part fails, 
        # but the trigger occurred.
