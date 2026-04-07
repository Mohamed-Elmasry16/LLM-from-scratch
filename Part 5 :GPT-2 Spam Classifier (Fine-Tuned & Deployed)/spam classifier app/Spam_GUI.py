import customtkinter as ctk
import threading
from classify import load_classifier, classify_text
import os

# Styling
COLOR_BG = "#1A1C1E"
COLOR_SIDEBAR = "#212529"
COLOR_ACCENT = "#3D444B"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spam_classifier.pth")
class SpamClassifierApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("AI Spam Sentinel - GPT-2 Classifier")
        self.geometry("900x600")
        ctk.set_appearance_mode("dark")

        # Load Model
        try:
            self.model, self.device = load_classifier(MODEL_PATH)
            self.status = "Online"
        except Exception as e:
            self.status = f"Error: {str(e)}"

        # UI Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="MODELS INFO", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.pack(pady=20)
        
        self.info_label = ctk.CTkLabel(self.sidebar, text=f"Status: {self.status}\nModel: GPT-2 Small\nClasses: Ham, Spam", justify="left")
        self.info_label.pack(padx=10, pady=10)

        # Main Content
        self.main_frame = ctk.CTkFrame(self, fg_color=COLOR_BG)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        self.label = ctk.CTkLabel(self.main_frame, text="Paste Email or Message Content Below:", font=ctk.CTkFont(size=16))
        self.label.pack(pady=(0, 10))

        self.input_box = ctk.CTkTextbox(self.main_frame, height=300, fg_color="#2B2B2B", font=("Consolas", 12))
        self.input_box.pack(fill="both", expand=True, padx=10, pady=10)

        self.btn_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x", pady=10)

        self.classify_btn = ctk.CTkButton(self.btn_frame, text="ANALYSIS MESSAGE", command=self.start_analysis, fg_color="#4A90E2", hover_color="#357ABD")
        self.classify_btn.pack(side="right", padx=10)

        # Result Display
        self.result_label = ctk.CTkLabel(self.main_frame, text="RESULT: WAITING...", font=ctk.CTkFont(size=24, weight="bold"))
        self.result_label.pack(pady=20)
        
        self.conf_label = ctk.CTkLabel(self.main_frame, text="Confidence: --%", font=ctk.CTkFont(size=14))
        self.conf_label.pack()

    def start_analysis(self):
        text = self.input_box.get("1.0", "end-1c")
        if not text.strip(): return
        
        self.classify_btn.configure(state="disabled", text="Processing...")
        self.result_label.configure(text="ANALYZING...", text_color="white")
        
        # Use threading to keep UI responsive
        threading.Thread(target=self.run_inference, args=(text,)).start()

    def run_inference(self, text):
        label, confidence = classify_text(self.model, self.device, text)
        self.after(0, lambda: self.update_ui(label, confidence))

    def update_ui(self, label, confidence):
        color = "#FF4C4C" if label == "SPAM" else "#4CC38A"
        self.result_label.configure(text=f"RESULT: {label}", text_color=color)
        self.conf_label.configure(text=f"Confidence Score: {confidence:.2f}%")
        self.classify_btn.configure(state="normal", text="ANALYSIS MESSAGE")

if __name__ == "__main__":
    app = SpamClassifierApp()
    app.mainloop()