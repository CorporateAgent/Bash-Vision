import os
import json
import clip
import torch
from PIL import Image
from termcolor import colored  # For colored terminal output

# Function to get the name of the last modified prediction folder
def get_last_modified_folder(base_path="runs/detect"):
    # List all folders in the base_path
    folders = [os.path.join(base_path, f) for f in os.listdir(base_path) if f.startswith("predict") and os.path.isdir(os.path.join(base_path, f))]
    
    if not folders:
        print("No prediction folders found.")
        return None
    
    # Get the most recently modified folder
    last_modified_folder = max(folders, key=os.path.getmtime)
    print(colored(f"Last modified prediction folder found: {last_modified_folder}", "green"))
    return last_modified_folder

# Load facets from a JSON file based on the category
def load_facets(category, facets_dir="data/results/facets"):
    json_path = os.path.join(facets_dir, f"{category}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as file:
            data = json.load(file)
            
            # Navigate through the JSON structure to get the facets
            facets = data.get('data', {}).get('facets', {}).get('facets', [])
            categorized_facets = {}
            for facet in facets:
                facet_name = facet['name']
                values = [{"name": value['name'], "quantity": value['quantity']} for value in facet.get('values', [])]
                categorized_facets[facet_name] = values
            return categorized_facets
    return {}

# Load the OpenAI-CLIP model from local directory
def load_clip_model():
    device = "cpu"  # Force CPU usage since you're on a Mac without CUDA
    model_name = "ViT-B/32"  # This is the model name that clip.load expects

    # Cache location for the CLIP model
    local_model_path = "/Users/lorenzop/Desktop/Bash Vision/data/models/OpenAI-CLIP"
    
    # Set the environment variable so that CLIP looks for models in your custom directory
    os.environ["TORCH_HOME"] = local_model_path
    
    # Load the model from the specified model name (not the direct path)
    model, preprocess = clip.load(model_name, device=device, jit=False)

    print(colored(f"Loaded CLIP model: {model_name} from {local_model_path}", "blue"))
    return model, preprocess, device

# Predict the top matching facets using OpenAI-CLIP and then select the one with sufficient quantity
def predict_with_clip(model, preprocess, device, image_path, facet_list):
    # Ensure at least one facet exists
    if not facet_list:
        print(colored("No facets available for prediction.", "red"))
        return [], None

    # Preprocess the image and tokenize the facet names
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([facet["name"] for facet in facet_list]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # Handle cases where there are fewer than 5 facets
    topk_limit = min(5, len(facet_list))
    top_indices = similarity.topk(topk_limit, dim=-1).indices.squeeze(0).tolist()
    top_facets = [(facet_list[i]["name"], similarity[0, i].item(), facet_list[i]["quantity"]) for i in top_indices]

    # Find the best facet that has a quantity > 5, else use the next highest
    best_facet = None
    for facet in top_facets:
        if facet[2] > 5:  # Check if the quantity is greater than 5
            best_facet = facet
            break

    # If no facet meets the quantity threshold, use the highest scoring facet
    if not best_facet:
        best_facet = top_facets[0]

    return top_facets, best_facet

def main():
    # Dictionary to store selected facets for each category
    selected_facets_by_category = {}
    # List to store all the category names
    category_list = []

    # Find the most recently modified YOLO prediction folder
    latest_folder = get_last_modified_folder()

    if not latest_folder:
        print(colored("No prediction folders found.", "red"))
        return

    # Navigate to the crops folder within the latest prediction
    crops_folder = os.path.join(latest_folder, "crops")
    print(colored(f"Checking for crops folder at: {crops_folder}", "yellow"))
    
    if not os.path.exists(crops_folder):
        print(colored(f"No crops folder found at: {crops_folder}", "red"))
        return

    print(colored(f"Processing crops in the folder: {crops_folder}", "green"))
    
    # Load the CLIP model
    clip_model, clip_preprocess, device = load_clip_model()

    # Process each category in the crops folder
    for category in os.listdir(crops_folder):
        category_folder = os.path.join(crops_folder, category)
        facets_dict = load_facets(category)  # Load facets, categorized by their type
        
        if not facets_dict:
            print(colored(f"No facets found for category: {category}", "red"))
            continue

        print(colored(f"\nRunning predictions for category: {category}", "magenta"))
        print(colored("="*50, "white"))  # Divider line

        # Initialize list to store selected facets for this category
        selected_facets_list = []

        # Process only the first image in the category folder
        processed_image = False  # Flag to check if we've already processed an image
        for image_file in os.listdir(category_folder):
            if image_file.endswith(".jpg") and not processed_image:
                image_path = os.path.join(category_folder, image_file)
                
                # Run inference for each facet type separately
                for facet_type, facet_values in facets_dict.items():
                    if facet_values:
                        try:
                            top_facets, best_facet = predict_with_clip(clip_model, clip_preprocess, device, image_path, facet_values)
                            
                            print(colored(f"\nImage: {image_file} | Facet Type: {facet_type}", "blue"))
                            print(colored(f"Top {len(top_facets)} Facets:", "blue"))
                            for facet_name, score, quantity in top_facets:
                                print(f"  - {facet_name} | Score: {colored(f'{score:.4f}', 'yellow')} | Quantity: {colored(quantity, 'yellow')}")
                            print(colored(f"Selected Facet: {best_facet[0]} | Score: {best_facet[1]:.4f} | Quantity: {best_facet[2]}", "green"))
                            print(colored("="*50, "white"))  # Divider line
                            
                            # Add the selected facet to the list for this category, including quantity
                            selected_facets_list.append({"facet_type": facet_type, "selected_facet": best_facet[0], "quantity": best_facet[2]})

                        except Exception as e:
                            print(colored(f"Error processing {facet_type} for {image_file}: {e}", "red"))

                processed_image = True  # Set the flag to True after processing one image
                break  # Break after processing the first image

        # Store the selected facets for this category in the dictionary
        selected_facets_by_category[category] = selected_facets_list
        # Add the category to the category list
        category_list.append(category)

    # Print the selected facets for each category
    print(colored("\nSelected Facets by Category:", "cyan"))
    for category, facets in selected_facets_by_category.items():
        print(f"Category: {colored(category, 'yellow')}")
        for entry in facets:
            print(f"  Facet Type: {colored(entry['facet_type'], 'yellow')}, Selected Facet: {colored(entry['selected_facet'], 'yellow')} (Quantity: {colored(entry['quantity'], 'yellow')})")
        print(colored("="*50, "white"))  # Divider line

    # Print the list of all categories
    print(colored("\nList of All Categories:", "cyan"))
    print(category_list)

    # Optionally, return the collected data for further use
    return selected_facets_by_category, category_list

if __name__ == "__main__":
    selected_facets, categories = main()