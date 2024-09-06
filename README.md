<!-- HEADER SECTION -->
<h1 align="center" style="font-family:Arial, sans-serif;">
  Bash Vision 📸
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/YOLO-8.1-green.svg" alt="YOLO Version">
  <img src="https://img.shields.io/badge/Streamlit-1.38-red.svg" alt="Streamlit Version">
  <img src="https://img.shields.io/badge/OpenAI-CLIP-purple.svg" alt="OpenAI CLIP">
</p>
<p align="center">
  <img src="/data/images/Preview.jpg" width="600px"/>
</p>
<!-- TABLE OF CONTENTS -->
<h2>Table of Contents</h2>

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Resources](#resources)
- [Future Enhancements](#future-enhancements)

<hr/>

<!-- INTRODUCTION SECTION -->
<h2 id="introduction" style=" font-family:Arial, sans-serif;">Introduction</h2>
<p style="font-family:Arial, sans-serif;">
Exploration on an AI-driven product discovery feature that allows customers to upload lifestyle images and receive product recommendations. This project leverages <b>YOLO (You Only Look Once)</b> for fast and accurate object detection and <b>OpenAI’s CLIP</b> model to match identified items to relevant product categories, brands, and other facets using multi-modal embeddings. By employing real-time data from the VTEX platform and a content-based filtering approach to streamline our customers shopping experience.

We achieve this through image-text search, where a query image is processed to return textual data. Objects of interest are represented as vectors in feature space, and the similarity between them is quantitatively measured. We've aimed to asses the suitability of AI models in production environments and understand the requirements for scaling this into a more robust feature. The project follows the key principle of leveraging open-source technology to build an ecosystem that Bash can harness for better product discovery and recommendation.
</p>

<!-- FEATURES SECTION -->
<h2 id="features" style="font-family:Arial, sans-serif;">Features</h2>
<ul style="font-family:Arial, sans-serif;">
  <li>Detect objects in the uploaded images, generating bounding boxes.</li>
  <li>Matches detected objects with relevant product facets (e.g., brand, color, material).</li>
  <li>Query VTEX in real-time to fetch relevant product data.</li>
  <li>Store uploads with labels to gain insight on customer preferences</li>

</ul>

<!-- INSTALLATION SECTION -->
<h2 id="installation" style="font-family:Arial, sans-serif;">Installation</h2>

<p style="font-family:Arial, sans-serif;">
  <b>Note:</b> Ensure you have <a href="https://pytorch.org/"> Pytorch </a> installed as well as YOLOv8 and OpenAI CLIP models available in the <code>data/models</code> directory. You can download the models from the: <a href="#resources">resources section</a>.
</p>

<pre><code style="font-family:monospace;">
# Clone the repository
git clone https://github.com/username/bash-vision.git

# Navigate into the project directory
cd bash-vision

# Install dependencies
pip install -r requirements.txt

# Optionally, create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Run the app using Streamlit
streamlit run app.py
</code></pre>

<p align="center">
  <img src="/data/images/Preview3.jpg" width="800px"/>
</p>

<!-- HOW IT WORKS SECTION -->
<h2 id="how-it-works" style="font-family:Arial, sans-serif;">How It Works</h2>

<p>Our approach is centered around a modular architecture that allows for more product categories to be included. The inference process is built around a modular architecture as well thanks to <a href="https://github.com/ultralytics)](https://github.com/ultralytics/ultralytics/tree/4673fae31d7f39902250982a33a67fa651cc7b5d">Ultralytics</a>, we allow for dynamic parameter adjustments, including confidence thresholds and Non-Maximum Suppression (NMS) settings. </p>
<p>Thankfully <a href="https://github.com/AILab-CVC/YOLO-World">YOLO-World v2</a> offers a robust object detection mechanism, detecting multiple objects in a single pass. Once detected, bounding boxes are cropped and fed into the CLIP model, which then uses its zero-shot learning capabilities to match the detected objects against product categories and facets without requiring task-specific training.</p>

<p>We strategically combine cosine similarity scores from CLIP with data from <a href="https://github.com/vtex-apps/store-graphql/">VTEX</a>. If the initial set of predictions results in insufficient product matches, the system intelligently drops the lowest-ranked facets (based on both prediction confidence and product quantity) and reruns the query, maximizing the likelihood of surfacing items that are not only relevant but also in stock. Since the text-to-image matching occurs within a shared multi-modal space, the precision of the recommendations is directly influenced by the accuracy facet labels within the catalog. Poorly defined or inconsistent textual data can degrade the model’s ability to match items accurately.</p>

<p align="center">
  <img src="/data/images/Preview2.png" width="800px"/>
</p>

<h3 style="font-family:Arial, sans-serif;"> Process: </h3>

<ul style="font-family:Arial, sans-serif;">
  <li><b>Object Detection:</b> Uploaded lifestyle images may contain multiple objects or background noise. Using <a href="https://github.com/ultralytics/yolov8" target="_blank">YOLO-World v2</a>, the app detects relevant product categories (e.g., Sneakers, Shirts) and crops out individual items for further analysis.</li><br>
  
  <li><b>Embedding Space:</b> Detected objects are represented as vectors in an embedding space, using <a href="https://openai.com/research/clip" target="_blank">OpenAI’s CLIP</a>. The embedding space is dynamically populated from product facets like brand, color, and material. The multi-modal architecture allows both image and text to query the same space.</li><br>
  
  <li><b>Product Quantity:</b> To ensure relevant product suggestions, we retrieve product quantities through facet data prioritizing facets with higher product availability, we ensure that recommended products are readily in stock and meet user expectations.</li><br>

  <li><b>Product Ranking:</b> Our methodology ranks facets based on a hybrid scoring system that blends product quantity and CLIP prediction scores. This ranking ensures that high-confidence predictions are balanced with the availability of products, delivering the best matches to users.</li><br>
  
  <li><b>VTEX Integration:</b> The app sends queries to <a href="https://developers.vtex.com/vtex-developer-docs/docs/guides/graphql-overview" target="_blank">VTEX’s GraphQL API</a> to fetch product details in real-time, ensuring up-to-date and accurate recommendations.</li><br>
</ul>

<p align="center">
  <img src="/data/images/Preview4.jpg" width="800px"/>
</p>

<h3 style="font-family:Arial, sans-serif;">Considerations:</h3>
<ul style="font-family:Arial, sans-serif;">

  <li><b>Garment detection:</b> While current object detection works well for general items, detecting specific garments can be challenging without lowering the confidence threshold. Fine-tuning or training the model on fashion-specific datasets could improve its ability to detect garments with high accuracy.</li>

  <li><b>Noise Filtering:</b> YOLO detects objects in the image but does not perform advanced noise filtering. Future improvements could explore image segmentation to further reduce irrelevant elements.</li>

  <li><b>Expand on object categories:</b> We've seen models online trained on fashion items, our model is still basic open source model not trained on apparel or garments.</li>
  
  <li><b>Performance in Production:</b> This application works in real-time, but scaling will require optimization for handling larger datasets, more products, and model fine-tuning for specific cases.</li>
  
  <li><b>Embedding Management:</b> The current setup dynamically generates embeddings per query, without storing them long-term. Managing stored embeddings for future scalability would be an area for further development.</li>
</ul>


<!-- PROJECT STRUCTURE SECTION -->
<h2 id="project-structure" style="font-family:Arial, sans-serif;">Project Structure</h2>

<pre><code style="font-family:monospace;">
Bash Vision/
│
├── data/                 
│   ├── images/                           # Image datasets utilized for UI
│   ├── models/                           # Storage for machine learning models (YOLOv8, CLIP)
│   └── results/                          # Directory for intermediary data
│       └── facets/                       # Product taxonomies (e.g., Sneakers)
│             └── Category-3.JSON         # List of product subcategories
│
├── runs/                 
│   ├── detect/                           # Results from YOLO's object detection
│   │   └── predict/                      # Outputs from YOLO inference, including bounding box metadata
│   └── Context.txt                       # Contextual file for understanding prediction data
│
├── src/
│   ├── api.py                            # Responsible for orchestrating GraphQL query generation 
│   ├── YOLO_inference.py                 # Object recognition algorithm leveraging YOLOv8's neural architecture
│   ├── CLIP_inference.py                 # Module executing similarity analysis between visual inputs and textual embeddings
│   └── Facet_index.py                    # Script to dynamically update and retrieve product classification indices 
│
├── .streamlit/                           # Streamlit's user interface
│   └── config.toml                       # theme settings for UI components
│
├── app.py                                # Core application 
├── requirements.txt                      # Dependency manifest 
└── README.md                             # Project documentation <br>
</code></pre>

<!-- RESOURCES SECTION -->
<h2 id="resources" style="font-family:Arial, sans-serif;">🔗 Resources</h2>
<ul style="font-family:Arial, sans-serif;">
  <li>
    <a href="https://docs.ultralytics.com/models/yolo-world/" target="_blank">
      YOLOv8 Documentation
    </a> - Learn more & download YOLO World, one of the most efficient and fast object detection models used in this project.
  </li>
  <li>
    <a href="https://openai.com/research/clip" target="_blank">
      OpenAI CLIP Documentation
    </a> - Explore OpenAI's CLIP model bridges images and text, allowing for accurate product-facet matching in Bash Vision.
  </li>
  <li>
    <a href="https://docs.streamlit.io/" target="_blank">
      Streamlit Documentation
    </a> - Understand how Streamlit makes it easy to build and deploy interactive data apps like Bash Vision.
  </li>
  <li>
    <a href="https://graphql.org/" target="_blank">
      GraphQL Documentation
    </a> - Discover the power of GraphQL, the API query language used to retrieve product data from the e-commerce platform.
  </li>
  <li>
    <a href="https://python.org" target="_blank">
      Python Documentation
    </a> - Python is the foundation of this project. Review its official documentation for more details on language features and libraries.
  </li>
</ul>

<!-- FUTURE ENHANCEMENTS SECTION -->
<h2 id="future-enhancements" style="font-family:Arial, sans-serif;">Future Enhancements</h2>
<p style="font-family:Arial, sans-serif;">
  Enhancements that can be made to improve performance & experience
</p>

<ul style="font-family:Arial, sans-serif;">
<li>📊 <b>Advanced Analytics:</b> Offering insights into user behavior, search patterns, and product interest to provide better recommendations.</li>
  <li>📦 <b>Real-time stock updates:</b> Fetching real-time product availability and stock levels from VTEX.</li>
  <li>💻 <b>Customizable model selection:</b> Allowing users to choose between different AI models for their search, such as faster models for quick results or more accurate ones for detailed analysis.</li>
  <li>⚡ <b>Faster Processing:</b> Optimizing model loading times and query speeds for an even more seamless experience.</li>
  <li>👗 <b>Improved Apparel Detection:</b> Fine-tuning the model on our fashion-specific datasets could improve its ability to detect our garment products with high accuracy.</li>
</ul>
</ul>

<!-- LICENSE SECTION -->
<h2 id="license" style="font-family:Arial, sans-serif;">License</h2>

<p align="center" style="font-family:Arial, sans-serif;">
  Made with ❤️ by Lo.
</p>
