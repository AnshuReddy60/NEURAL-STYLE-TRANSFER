# NEURAL-STYLE-TRANSFER

**COMPANY**: CODTECH IT SOLUTIONS

**NAME**: ANSHU VEERAMALLA

**INTERN ID**: CODF269

**DOMAIN**: AIML(ARTIFICIAL INTELLIGENCE MARKUP LANGUAGE)

**DURATION**: 1 MONTH

**MENTOR**: NEELA SANTOSH

## DESCRIPTION

# Neural Style Transfer with PyTorch

This project implements **Neural Style Transfer (NST)**, a technique in deep learning where the **content** of one image is transferred onto the **style** of another. The result is a new image that combines the content of the first image with the artistic style of the second. This technique is based on **Convolutional Neural Networks (CNNs)** and can produce visually striking outputs, often used in artistic applications. In this project, we use a **pre-trained VGG19 model** to extract features from both content and style images, and optimize the resulting image to minimize the differences in content and style.

### **Overview**

The core idea behind **Neural Style Transfer** is to take a content image (e.g., a photograph) and a style image (e.g., a painting), and apply the style of the second image to the content of the first. This is done by extracting **feature maps** using a pre-trained CNN and then optimizing a new image to match the content features of the content image and the style features of the style image.

This implementation uses **PyTorch** and the **VGG19 model**, a popular CNN architecture, to perform the style transfer.

### **Key Features**

- **Content and Style Loss**: The project defines custom loss functions to calculate the difference between content features (extracted from the content image) and style features (extracted from the style image). The model uses these loss functions to iteratively adjust the generated image.
  
- **Gram Matrix**: The Gram matrix is used to calculate the style loss. It captures correlations between different filter responses in the feature map, which helps capture the texture of the image.

- **VGG19 Pre-trained Model**: A pre-trained **VGG19** model is used to extract deep features from both content and style images. It is a common practice in NST since the VGG19 model is trained on large-scale image datasets and can capture hierarchical image features.

- **Gradient Descent Optimization**: The model uses gradient descent (through the Adam optimizer) to minimize the loss, gradually transforming a randomly initialized image into one that resembles the content image but in the style of the style image.

- **Adjustable Weights**: The weights for the style and content losses can be adjusted, allowing you to control the balance between preserving the content and the style.

### **How It Works**

1. **Image Loading**: The images are loaded and resized to a fixed size (512x512 pixels). They are then converted into tensors, which are the format used by PyTorch models.

2. **Feature Extraction**: The VGG19 model is used to extract features from both the content and style images. The model is stripped of the final classification layers, leaving the convolutional layers that capture the image features.

3. **Loss Calculation**: 
   - The **content loss** is calculated as the Mean Squared Error (MSE) between the features of the generated image and the content image.
   - The **style loss** is calculated using the Gram matrix of the feature maps and measuring the difference between the style of the generated image and the style image.

4. **Optimization**: The image is iteratively optimized using the **Adam optimizer**. The goal is to minimize the combined content and style losses, thereby adjusting the image to resemble the content image but with the style of the style image.

5. **Output**: After the optimization process, the final image is saved and displayed.

### **Technologies Used**

- **PyTorch**: The main framework for building and training the model.
- **VGG19**: A pre-trained CNN model used to extract features.
- **PIL**: Python Imaging Library (PIL) is used for image manipulation (loading, saving, and converting between formats).
- **TorchVision**: Provides pre-trained models like VGG19 and image transformation utilities.

### **How to Run the Application**

#### **1. Install Required Libraries**

To run the neural style transfer script, first install the required dependencies:

```bash
pip install torch torchvision pillow
```

#### **2. Download or Clone the Repository**

You can clone the repository or download the project files. Make sure you have a **content image** (the image you want to apply the style to) and a **style image** (the artwork or image from which the style will be taken).

#### **3. Run the Style Transfer**

Once the dependencies are installed, and the images are ready, you can run the style transfer script by executing:

```bash
python neural_style_transfer.py
```

This will perform the style transfer and save the resulting image in the specified output path.

#### **4. Adjust Parameters**

You can modify the script to use different content and style images, or adjust the **style weight** and **content weight** parameters to fine-tune the output.

### **Conclusion**

This **Neural Style Transfer** project demonstrates how deep learning models can be used for artistic applications. Using a pre-trained VGG19 model, it combines content and style from two images to create a new, artistic image that merges both. The project illustrates the potential of **deep learning** for creative and artistic purposes, and serves as a great introduction to using **PyTorch** for image-based tasks.

## OUTPUT

![Image](https://github.com/user-attachments/assets/9fd5e5ea-c234-4d31-8140-2c3cf46af67e)
