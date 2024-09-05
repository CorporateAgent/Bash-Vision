import requests
from termcolor import colored
import streamlit as st

# Set the correct GraphQL endpoint
GRAPHQL_ENDPOINT = "https://thefoschini.myvtex.com/_v/segment/graphql/v1/"

# Function to generate the GraphQL query for a given category and facets
def generate_graphql_query(category, selected_facets):
    facets_str = ",\n".join([f'{{key:"{facet["facet_type"]}", value:"{facet["selected_facet"]}"}}' for facet in selected_facets])
    query = f"""
    query Request {{
      productSearch( 
        hideUnavailableItems:true,
        selectedFacets:[
        {{key:"category-3", value:"{category}"}},
        {facets_str}] )
      @context(provider: "vtex.search-graphql@0.62.0") {{
        products {{
          productName
          link
          items(filter:FIRST_AVAILABLE) {{
            images{{imageUrl}}
          }}
        }}
      }}
    }}
    """
    return query

# Function to execute the GraphQL query and return the results
def execute_graphql_query(query):
    response = requests.post(GRAPHQL_ENDPOINT, json={'query': query})
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"GraphQL query failed with status code {response.status_code}")
        return None

# Function to sort facets by product quantity and remove the lowest when necessary
def sort_and_adjust_facets(category, facets_list):
    facets_list.sort(key=lambda facet: facet['quantity'], reverse=True)

    while facets_list:
        query = generate_graphql_query(category, facets_list)
        
        # Print only the selected facets portion of the query, with colored key-value pairs
        print(colored(f"\nSelected facets for category '{category}':", "cyan"))
        for facet in facets_list:
            key = colored(f'key:"{facet["facet_type"]}"', "yellow")
            value = colored(f'value:"{facet["selected_facet"]}"', "yellow")
            print(f"  {{{key}, {value}}}")

        graphql_results = execute_graphql_query(query)

        if graphql_results:
            products = graphql_results['data']['productSearch']['products']
            if len(products) >= 5:
                print(colored("\nProduct Links:", "cyan"))
                for product in products:
                    link = f"https://bash.com{product.get('link', '#')}"
                    print(colored(link, "blue", attrs=["bold"]))
                return products

        facets_list.pop()

    return []