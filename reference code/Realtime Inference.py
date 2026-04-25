import torch
# I-import mo yung dalawang magkahiwalay na architecture mo
from cnn_model import BassCNN       # Yung PyTorch Model mo
from decoder import BassTranscriptionDecoder # Yung ginawa mo kaninang HMM class

def run_realtime_transcription():
    # 1. I-LOAD ANG MGA MAKINA (Initialization)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Makina A: Ang AI (Loaded to GPU/CPU)
    cnn_model = BassCNN().to(device)
    cnn_model.load_state_dict(torch.load("bass_weights.pth"))
    cnn_model.eval() # STRICT: Naka-eval mode para walang backprop
    
    # Makina B: Ang Post-Processor (Tatakbo lang sa CPU via NumPy)
    decoder = BassTranscriptionDecoder(onset_threshold=0.6, offset_threshold=0.6)
    
    print("Ready to rock, beh. Start playing...")
    
    # 2. ANG CONVEYOR BELT (Real-Time Audio Loop)
    while True:
        # A. Kumuha ng 50ms audio frame mula sa mic (Pseudo-code)
        audio_frame = get_audio_from_mic() 
        spectrogram = preprocess_to_mel(audio_frame).to(device)
        
        # B. IPASOK SA CNN (Feature Extraction)
        with torch.no_grad(): # Patayin ang gradients para mabilis sa Jetson Nano
            # Ang output ng CNN mo ay listahan ng tensors
            cnn_predictions = cnn_model(spectrogram) 
            
        # C. IPASA SA DECODER (Ang Data Hand-off)
        # Dito papasok yung output ng PyTorch papunta sa NumPy/HMM logic mo!
        note_event = decoder.decode_cnn_output(cnn_predictions)
        
        # D. I-PRINT SA SCREEN KUNG MAY TUMUNOG
        if note_event:
            if note_event["event"] == "ONSET":
                print(f"🎸 PITIK: {note_event['decoded_string']} | {note_event['decoded_fret']} | {note_event['decoded_pitch']}")
            elif note_event["event"] == "OFFSET":
                print(f"🔇 BITAW: {note_event['decoded_string']}")

if __name__ == "__main__":
    run_realtime_transcription()