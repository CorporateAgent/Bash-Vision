import requests
import json
import time
import os

# Load the categories from the JSON file
with open("data/results/category-3.json", 'r') as file:
    data = json.load(file)
    categories = data["category-3"]  # Extract the list of categories

# GraphQL query template with properly escaped curly braces
QUERY_TEMPLATE = """
query Request {{
  facets(hideUnavailableItems: true, selectedFacets: {{key: "category-3", value: "{category}"}}) @context(provider: "vtex.search-graphql@0.62.0") {{
    facets {{
      name
      quantity
      values {{
        name
        quantity
      }}
    }}
  }}
}}
"""

# Where to save the JSON files
save_folder = "data/results/facets"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Facets to exclude
excluded_facets = {"Price", "Store", "Size", "Department", "Category", "Subcategory","Sport", "Bags & wallets","Gemstone","Necklace"}

# Function to filter the JSON response
def filter_response(data):
    filtered_facets = []
    for facet in data["data"]["facets"]["facets"]:
        if facet["name"] not in excluded_facets:
            filtered_facets.append(facet)
    
    data["data"]["facets"]["facets"] = filtered_facets
    return data

# Loop over each category
for category in categories:
    # Insert the actual category value into the query
    query = QUERY_TEMPLATE.format(category=category)
    
    # Print the actual query being sent for debugging
    print(f"Sending query for category: {category}")
    
    # Send the request to the server
    response = requests.post("https://thefoschini.myvtex.com/_v/segment/graphql/v1/", json={'query': query})
    
    if response.status_code == 200:
        response_json = response.json()
        
        # Print the raw response for debugging
        print("Response received:")
        
        # Filter the response to remove excluded facets
        filtered_response = filter_response(response_json)
        
        # Save the filtered result as a JSON file
        with open(f"{save_folder}/{category}.json", 'w') as json_file:
            json.dump(filtered_response, json_file, indent=4)
    else:
        print(f"Failed to fetch data for {category}: {response.status_code}")
    
    # Wait 5 seconds before the next request
    time.sleep(5)