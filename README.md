
# Fashion Recommender System Using Transfer Learning

This repository contains the code for my recreation of the **"Image Based Recommender System using Transfer Learning"** paper by Nikhil Kumar Singh and Abhimanyu Kumar from NIT Uttarakhand. The project implements a fashion recommender system using image embeddings generated through the pre-trained VGG16 model and transfer learning. The system recommends visually similar fashion items based solely on the images users interact with, leveraging cosine similarity for item retrieval.

## Differences from the Original Paper
While the core methodology remains similar to the paper, there are some notable differences in this implementation:
- **Dataset Used**: The original paper utilized the high-resolution **Fashion Product Images Dataset** (2400x1600 resolution), while this recreation uses the smaller version of the dataset, titled **Fashion Product Images (Small)**, available on Kaggle. The small version contains lower-resolution images, but the overall structure remains the same, making it suitable for image-based recommendation experiments on a smaller scale.
- **Scope and Resources**: Due to resource constraints, I limited the dataset to 3000 images for the purposes of feature extraction and recommendation to avoid system crashes during compilation. The original work did not have such limitations.
- **Implementation Environment**: This recreation was built using Google Colab, leveraging free GPU resources for model training and inference.

## Dataset: Fashion Product Images (Small)

The **Fashion Product Images (Small)** dataset consists of professionally shot product images along with manually entered label attributes for cataloging. The dataset is ideal for exploring various machine learning models such as image classification and recommendation systems.

### Dataset Details:
- **Images**: The dataset contains images of fashion products identified by an ID, which can be mapped via the `styles.csv` file.
- **Attributes**: Each product is associated with attributes like master category, subcategory, gender, and display names, also found in `styles.csv`.
- **Inspiration**: 
    - Train an image classifier using the `masterCategory` column and a convolutional neural network.
    - Alternatively, classify based on product descriptions from the `styles.json` files.
    - Explore multi-label classification by predicting other product category labels.

For more information on the dataset, check the following links:
- [Fashion Product Images (Small) on Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- [High-Resolution Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

## Approach

1. **Transfer Learning**: The pre-trained VGG16 model is used to extract feature embeddings from each image, utilizing weights trained on the ImageNet dataset.
2. **Cosine Similarity**: The system computes cosine similarity between the image embeddings to identify visually similar items. Top K similar items are then recommended based on this similarity measure.
3. **Feature Extraction**: Images are resized to 224x224 pixels as required by the VGG16 architecture. The `fc2` layer of VGG16 is used to generate the feature embeddings for each image.
4. **Recommendation**: Given an image input, the system retrieves the top 5 most similar images and displays them along with their similarity scores.

## Getting Started

### Prerequisites
- Python 3.x
- TensorFlow/Keras
- Matplotlib
- NumPy
- scikit-learn
- pandas

### How to Run

1. Install the required libraries:
   ```bash
   pip install tensorflow keras matplotlib numpy scikit-learn pandas
   ```

2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) and extract it into your working directory.

3. Run the notebook or Python script to load the images, process them through the VGG16 model, and retrieve recommendations based on image similarity.

## Conclusion
This recreation successfully demonstrates how transfer learning can be applied to a fashion recommender system using image similarity. Despite differences in the dataset and computational resources, the system performs well in providing recommendations based on visual features.

Feel free to explore and modify the code for your own experiments!

## References
- Singh, N. K., & Kumar, A. (2022). Image Based Recommender System using Transfer Learning. *2022 2nd International Conference on Emerging Frontiers in Electrical and Electronic Technologies (ICEFEET)*.
- [Fashion Product Images (Small) Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
