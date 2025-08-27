#Handwritten Digit Recognition (CNN + PyTorch + Streamlit)

This project recognizes handwritten digits (0–9) using a **Convolutional Neural Network** trained on MNIST dataset.

##How to Run Locally
```bash
pip install -r requirements.txt
python train_model.py   # trains and saves model as mnist_cnn.pth
streamlit run app.py    # launches web app
