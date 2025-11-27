import streamlit as st
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F


class DeepfakeEnsemble(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeEnsemble, self).__init__()

        self.effnet = timm.create_model('efficientnet_b4', pretrained=False, num_classes=0)
        self.effnet_head = nn.Sequential(
            nn.Linear(1792, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes) 
        )


        self.xception = timm.create_model('xception', pretrained=False, num_classes=0)
        self.xception_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        eff_features = self.effnet(x)
        eff_logits = self.effnet_head(eff_features)
        
        xcp_features = self.xception(x)
        xcp_logits = self.xception_head(xcp_features)
        
        avg_logits = (eff_logits + xcp_logits) / 2.0
        return avg_logits


st.set_page_config(page_title="Deepfake Detector")


MODEL_PATH = 'best_deepfake_ensemble.pth'  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
IMG_SIZE = 299

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


@st.cache_resource 
def load_trained_model():
    try:
        model = DeepfakeEnsemble(num_classes=2)
        
        state_dict = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
        model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval() 
        return model
    except FileNotFoundError:
        st.error(f"Model file '{MODEL_PATH}' not found. Please place it in the same directory.")
        return None


st.title("Deepfake Detection Ensemble")
st.write("Upload an image to detect if it is **Real** or **Fake** using EfficientNet + Xception.")

model = load_trained_model()

if model is not None:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button('Analyze Image'):
            with st.spinner('Analyzing for artifacts...'):
                img_tensor = preprocess(image).unsqueeze(0).to(DEVICE) 

                with torch.no_grad():
                    logits = model(img_tensor)
                    probabilities = F.softmax(logits, dim=1)
                
                fake_prob = probabilities[0][0].item()
                real_prob = probabilities[0][1].item()
                
                st.divider()
                if fake_prob > real_prob:
                    st.error(f"**Verdict: FAKE**")
                    st.progress(fake_prob, text=f"Confidence: {fake_prob*100:.2f}%")
                else:
                    st.success(f"**Verdict: REAL**")
                    st.progress(real_prob, text=f"Confidence: {real_prob*100:.2f}%")
                
                with st.expander("View Raw Probabilities"):
                    st.write(f"Fake Probability: {fake_prob:.4f}")
                    st.write(f"Real Probability: {real_prob:.4f}")