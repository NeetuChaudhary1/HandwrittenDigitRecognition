import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st
from model import CNN
from streamlit_drawable_canvas import st_canvas

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()

st.title("Handwritten Digit Recognition")
st.write("Upload a digit image OR draw a digit below:")

uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])


st.subheader("Draw a digit here")
canvas_result = st_canvas(
    fill_color="black",  
    stroke_width=10,
    stroke_color="white",  
    background_color="black",
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas",
)

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        probs = F.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred, probs.cpu().numpy()[0]

if uploaded_file:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="Uploaded Image", width=150)
    pred, probs = predict(img)
    st.subheader(f"Prediction: {pred}")
    st.bar_chart(probs)

if canvas_result.image_data is not None:
    drawn_img = Image.fromarray((canvas_result.image_data[:, :, 0] > 0).astype("uint8") * 255)  
    drawn_img = drawn_img.convert("L")  
    if drawn_img.getbbox():  
        st.image(drawn_img.resize((100,100)), caption="Your Drawing", width=100)
        pred, probs = predict(drawn_img)
        st.subheader(f"Prediction: {pred}")
        st.bar_chart(probs)
