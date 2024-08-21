import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import keras
import random
import tensorflow as tf

# Load model and dataset
model = keras.models.load_model('MNIST.h5')
feature_model = keras.models.Model(model.inputs, [layer.output for layer in model.layers])
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test = x_test / 255.0

# Streamlit app
st.title('Neural Network Visualizer')
st.sidebar.title('Input Image')
index = random.randint(0, 10000)

if st.sidebar.button('Get Random Prediction'):
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    preds = feature_model.predict(image_arr)

    # Display input image
    st.sidebar.image(np.reshape(image, (28, 28)), width=250, caption="Input Image")
    st.sidebar.markdown(f"Original Label: {y_test[index]}")

    # Visualize each layer
    for i, layer_output in enumerate(preds):
        st.subheader(f"Layer {i+1}: {model.layers[i].__class__.__name__}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 2))
            
            # Reshape the output for visualization
            if len(layer_output.shape) == 4:  # Convolutional layer
                layer_output_2d = np.squeeze(layer_output).transpose(2, 0, 1).reshape(layer_output.shape[-1], -1)
            elif len(layer_output.shape) == 2:  # Dense layer
                layer_output_2d = layer_output.T
            else:
                layer_output_2d = layer_output.squeeze()
            
            im = ax.imshow(layer_output_2d, aspect='auto', cmap='viridis')
            ax.set_xlabel('Neuron/Feature')
            ax.set_ylabel('Sample/Spatial')
            plt.colorbar(im)
            st.pyplot(fig)
        
        with col2:
            st.write(f"Shape: {layer_output.shape}")
            st.write(f"Min: {layer_output.min():.4f}")
            st.write(f"Max: {layer_output.max():.4f}")
            st.write(f"Mean: {layer_output.mean():.4f}")
            st.write(f"Std: {layer_output.std():.4f}")

    # Display final prediction
    final_layer = preds[-1][0]
    st.subheader("Final Layer (Output)")
    fig, ax = plt.subplots()
    ax.bar(range(10), final_layer)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Probability')
    ax.set_xticks(range(10))
    st.pyplot(fig)

    predicted_digit = np.argmax(final_layer)
    st.subheader(f"Model Prediction: {predicted_digit}")
    st.subheader(f"Confidence: {final_layer[predicted_digit]:.4f}")

st.sidebar.markdown("### *Made by: Divya K*")