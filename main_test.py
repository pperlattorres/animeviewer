from animeflv import AnimeFLV, EpisodeInfo, AnimeInfo, AnimeFLVParseError
with AnimeFLV() as api:
    try:
        # Get detailed information about an anime using its ID
        print("Ejecutando el script...")
        results = api.search("fire force")
        
        # Output the first result
        if results:
            print(f"Title: {results[0].title}")
            print(f"ID: {results[0].id}")
            print(f"Type: {results[0].type}")
    except AnimeFLVParseError as e:
        # Handle parsing error
        print(f"Error parsing response: {e}")
    except Exception as e:
        # Handle other exceptions
        print(f"Unexpected error: {e}")