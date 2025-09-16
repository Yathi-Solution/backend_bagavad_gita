import os
from dotenv import load_dotenv
from supabase import create_client, Client


load_dotenv()


def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment")
    return create_client(url, key)


# Singleton client
supabase: Client = get_supabase_client()