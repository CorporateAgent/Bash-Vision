import streamlit as st
import os
import threading  # For the lock mechanism
from src.YOLO_inference import perform_inference, log_results
from src.CLIP_inference import main as clip_inference
from src.api import sort_and_adjust_facets  # Importing from the api.py file

# Global lock to control access
processing_lock = threading.Lock()

# Set Streamlit page layout to wide
st.set_page_config(layout="wide")

# Split the layout into two columns (with a 1:2 ratio to give more space to results)
col1, col2 = st.columns([1, 2])
col2.title("AI-Powered Image Search")

# Placeholder for the status message in col1
status_placeholder = col2.empty()
status_placeholder.write("✨ Let’s find your style! Upload a photo, and we’ll search for the best matches just for you!")

# Placeholder for image display in col1 (to replace original image with processed one)
image_placeholder = col1.empty()

# Image uploader
uploaded_file = col1.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Try to acquire the lock
    if processing_lock.acquire(blocking=False):
        try:
            # Overwrite the placeholder message with the new status update
            status_placeholder.write("Processing your request, please wait...")

            # Save uploaded image to the designated folder
            image_path = os.path.join("data/images", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_placeholder.image(image_path, caption="Uploading Image...", use_column_width=True)  # Display original image
            status_placeholder.write("Image saved for processing.")

            # Perform YOLO inference on the uploaded image
            status_placeholder.write("🔍 Spotting the styles in your image… One moment!")
            results = perform_inference("data/models/YOLO-World/YOLOv8x Worldv2.pt", image_path)
            log_results(results)

            # Once YOLO inference is done, replace the uploaded image with the processed image
            processed_image_dir = os.path.join("runs/detect")  # Folder where YOLO stores results
            latest_predict_folder = sorted([f for f in os.listdir(processed_image_dir) if f.startswith("predict")], key=lambda x: os.path.getctime(os.path.join(processed_image_dir, x)))[-1]
            processed_image_path = os.path.join(processed_image_dir, latest_predict_folder, uploaded_file.name)  # YOLO keeps the same file name for the processed image

            if os.path.exists(processed_image_path):
                # Replace the uploaded image with the YOLO-processed image with bounding boxes
                image_placeholder.image(processed_image_path, use_column_width=True)
            else:
                status_placeholder.write("Processed image not found, showing original image.")

            status_placeholder.write("YOLO inference complete.")
            
            # Trigger CLIP inference on the detected objects
            status_placeholder.write("Finding the perfect pieces—matching with thousands of fashion options.✨")
            selected_facets, categories = clip_inference()
            status_placeholder.write("🤖 Putting together your look… Hang tight!")


            # After completion of CLIP inference, display results in col2
            if categories:
                for category in categories:
                    facets_list = selected_facets[category]

                    # Sort facets and adjust until we get at least 5 products
                    products = sort_and_adjust_facets(category, facets_list)
                    

                    if products:
                        
                        # col2.write(f"{category}")

                        # Display product images in a grid format (max 4 per category)
                        max_products = min(4, len(products))  # Limit to 4 products per category

                        # Create a grid of 4 columns
                        cols = col2.columns(4)  # 4 columns

                        # Loop through each product and display it in one of the 4 columns
                        for i, product in enumerate(products[:4]):  # Only show 4 products
                            with cols[i % 4]:  # Place each product in a column
                                if product['items'][0]['images']:
                                    st.image(product['items'][0]['images'][0]['imageUrl'], width=160)  # Adjust image width to 200px
                                    st.caption(product['productName'])
                                product_link = f"https://bash.com{product.get('link', '#')}"
                                st.button(f"[Shop on Bash]({product_link})")
                    else:
                        col2.write(f"No sufficient products found for category: {category}.")
            else:
                col2.write("No categories detected from YOLO inference.")
            
            # Once processing is done, you can replace the status with "Processing Complete" or similar message.
            status_placeholder.write("Done! ✨ Check out your personalized recommendations.")
        
        finally:
            # Release the lock after processing is complete
            processing_lock.release()
    else:
        # If another user is processing, show this message
        status_placeholder.write("Another user is currently processing a request. Please wait and try again shortly.")