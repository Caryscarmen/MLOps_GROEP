
import torch

import sys

import os



# Zorg dat de 'src' map gevonden kan worden voor imports

sys.path.append(os.path.abspath('src'))



from ml_core.models.mlp import MLP



def run_inference(model_path):

    # 1. Initialiseer de model-architectuur

    # Let op: vul de juiste dimensies in als je MLP die verwacht (bijv. input_dim)

    model = MLP() 

    

    # 2. Laad de getrainde gewichten

    if os.path.exists(model_path):

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        model.load_state_dict(checkpoint)

        model.eval()

        print(f"✅ Model succesvol geladen van {model_path}")

    else:

        print(f"❌ Fout: Kan modelbestand niet vinden op {model_path}")

        return



    # 3. Maak een nep-afbeelding (dummy data) voor de test

    # Voor een MLP moet de input vaak 'plat' zijn (96*96*3)

    dummy_input = torch.randn(1, 3 * 96 * 96) 

    

    with torch.no_grad():

        output = model(dummy_input)

        # Gebruik sigmoid voor binaire classificatie (tumor/geen tumor)

        prediction = torch.sigmoid(output).item()

        

    print(f"Voorspelde kans: {prediction:.4f}")

    print("Resultaat: " + ("Tumor" if prediction > 0.5 else "Gezond"))



if __name__ == "__main__":

    run_inference("models/best_model.pt")

