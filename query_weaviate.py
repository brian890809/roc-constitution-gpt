import weaviate
import os
from dotenv import load_dotenv

# Load your .env file
load_dotenv()

# Get API key and ensure it's not None
api_key = os.getenv("WEAVIATE_API_KEY")
if not api_key:
    raise ValueError("WEAVIATE_API_KEY environment variable is not set")

# Get Weaviate URL
weaviate_url = os.getenv("WEAVIATE_URL")
if not weaviate_url:
    raise ValueError("WEAVIATE_URL environment variable is not set")

# Initialize the client with the current API syntax
client = weaviate.connect_to_weaviate_cloud(
    weaviate_url,
    auth_credentials=weaviate.auth.AuthApiKey(api_key=api_key),
)
entries = client.collections.get("ROC_Constitution_BG3_M3")
response = entries.query.fetch_objects(
    limit=10,
)

for o in response.objects:
    print(o.uuid)