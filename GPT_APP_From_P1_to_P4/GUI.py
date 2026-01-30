import customtkinter as ctk
import torch
from GPT2_Archeticture import GPTModel
import tiktoken
import os
import threading
from genrate import generate

# ==========================================
# 1. BACKEND LOGIC (Unchanged)
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "GPT2_124M_torch.pth")

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    print("✅ Found model at:", MODEL_PATH)
    GPT = GPTModel(GPT_CONFIG_124M)
    GPT.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    GPT.to(device)
    GPT.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    MODEL_LOADED = True
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False

def generate_text_logic(prompt, max_new_tokens, temperature, top_k):
    if not MODEL_LOADED:
        return "Error: Model not loaded."
    try:
        idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        generated = generate(
            GPT, idx, max_new_tokens=max_new_tokens, context_size=1024, 
            temperature=temperature, top_k=top_k
        )
        return tokenizer.decode(generated.squeeze().tolist())
    except Exception as e:
        return f"Error: {str(e)}"

# ==========================================
# 2. MODERN UI DESIGN
# ==========================================

# Custom Colors
COLOR_BG = "#1a1a1a"          # Very Dark Grey (Main Window)
COLOR_SIDEBAR = "#252526"     # VS Code Sidebar Grey
COLOR_CHAT_BG = "#1e1e1e"     # Chat Background
COLOR_ACCENT = "#007acc"      # Modern Blue
COLOR_TEXT_BOX = "#333333"    # Input box background

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class ModernGPTApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Window Setup ---
        self.title("GPT-2 Neural Studio")
        self.geometry("1100x750")
        self.configure(fg_color=COLOR_BG)

        # --- Grid Layout ---
        self.grid_columnconfigure(1, weight=1) # Main chat area expands
        self.grid_rowconfigure(0, weight=1)

        # ================= LEFT SIDEBAR (Control Panel) =================
        self.sidebar = ctk.CTkFrame(self, width=280, fg_color=COLOR_SIDEBAR, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False) # Force width

        # Logo / Header
        self.logo = ctk.CTkLabel(self.sidebar, text="GPT-2 MINI", font=("Montserrat", 24, "bold"), text_color="white")
        self.logo.pack(pady=(30, 20))

        # Separator
        ctk.CTkFrame(self.sidebar, height=2, fg_color="#3e3e42").pack(fill="x", padx=20, pady=10)

        # SETTINGS GROUP
        self.settings_label = ctk.CTkLabel(self.sidebar, text="MODEL PARAMETERS", font=("Roboto", 12, "bold"), text_color="gray")
        self.settings_label.pack(anchor="w", padx=25, pady=(20, 10))

        # 1. Max Tokens
        self.create_slider_group("Max Tokens", 10, 200, 50, self.update_tokens)
        
        # 2. Temperature
        self.create_slider_group("Temperature", 0.1, 1.0, 0.7, self.update_temp)
        
        # 3. Top-K
        self.create_slider_group("Top-K", 10, 100, 50, self.update_topk)

        # Spacer to push buttons down
        self.sidebar_spacer = ctk.CTkLabel(self.sidebar, text="")
        self.sidebar_spacer.pack(expand=True)

        # Clear Button
        self.clear_btn = ctk.CTkButton(
            self.sidebar, text="Clear History", 
            fg_color="transparent", border_width=1, border_color="gray",
            hover_color="#c42b1c", height=40,
            command=self.clear_chat
        )
        self.clear_btn.pack(fill="x", padx=20, pady=20)

        # ================= RIGHT MAIN AREA (Chat) =================
        self.main_area = ctk.CTkFrame(self, fg_color=COLOR_BG)
        self.main_area.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        self.main_area.grid_rowconfigure(1, weight=1) # Chat log expands
        self.main_area.grid_columnconfigure(0, weight=1)

        # 1. Top Header
        self.header = ctk.CTkFrame(self.main_area, height=60, fg_color=COLOR_BG)
        self.header.grid(row=0, column=0, sticky="ew")
        
        self.status_indicator = ctk.CTkLabel(self.header, text="● Online", text_color="#4CC38A", font=("Roboto", 14, "bold"))
        self.status_indicator.pack(side="left", padx=20, pady=15)
        
        self.copy_btn = ctk.CTkButton(self.header, text="Copy Output", width=100, fg_color="#3e3e42", command=self.copy_text)
        self.copy_btn.pack(side="right", padx=20, pady=15)

        # 2. Chat Display (The "Screen")
        self.chat_display = ctk.CTkTextbox(
            self.main_area, 
            fg_color=COLOR_CHAT_BG, 
            text_color="#e0e0e0",
            font=("Consolas", 15), 
            corner_radius=15,
            wrap="word",
            border_spacing=20
        )
        self.chat_display.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.chat_display.insert("0.0", "SYSTEM: Ready. Initialize prompt sequence...\n\n")
        self.chat_display.configure(state="disabled")

        # 3. Input Area (The "Capsule")
        self.input_container = ctk.CTkFrame(self.main_area, fg_color=COLOR_BG, height=80)
        self.input_container.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 20))
        self.input_container.grid_columnconfigure(0, weight=1)

        # Input Box
        self.input_box = ctk.CTkEntry(
            self.input_container, 
            placeholder_text="Type your message here...",
            height=50,
            corner_radius=25, # Fully rounded
            fg_color=COLOR_TEXT_BOX,
            border_width=0,
            font=("Roboto", 14)
        )
        self.input_box.grid(row=0, column=0, sticky="ew", padx=(0, 15))
        self.input_box.bind("<Return>", lambda event: self.start_generation())

        # Generate Button (Icon style)
        self.send_btn = ctk.CTkButton(
            self.input_container, 
            text="➤", 
            width=50, height=50, 
            corner_radius=25, # Circle button
            fg_color=COLOR_ACCENT,
            font=("Arial", 20),
            command=self.start_generation
        )
        self.send_btn.grid(row=0, column=1)

        # Init variables for sliders
        self.val_tokens = 50
        self.val_temp = 0.7
        self.val_topk = 50

    # --- Helper to create pretty sliders ---
    def create_slider_group(self, title, min_val, max_val, default, cmd):
        frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        frame.pack(fill="x", padx=20, pady=10)
        
        label_row = ctk.CTkFrame(frame, fg_color="transparent")
        label_row.pack(fill="x")
        
        ctk.CTkLabel(label_row, text=title, font=("Roboto", 12)).pack(side="left")
        val_label = ctk.CTkLabel(label_row, text=str(default), font=("Roboto", 12, "bold"), text_color=COLOR_ACCENT)
        val_label.pack(side="right")
        
        slider = ctk.CTkSlider(
            frame, from_=min_val, to=max_val, 
            number_of_steps=20, 
            progress_color=COLOR_ACCENT, 
            button_color="white", 
            button_hover_color=COLOR_ACCENT,
            command=lambda v: [cmd(v), val_label.configure(text=f"{v:.0f}" if v > 1 else f"{v:.2f}")]
        )
        slider.set(default)
        slider.pack(fill="x", pady=(5, 0))
        return slider

    # --- Logic Handlers ---
    def update_tokens(self, val): self.val_tokens = int(val)
    def update_temp(self, val): self.val_temp = val
    def update_topk(self, val): self.val_topk = int(val)

    def clear_chat(self):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("0.0", "end")
        self.chat_display.insert("0.0", "SYSTEM: Context cleared.\n\n")
        self.chat_display.configure(state="disabled")

    def copy_text(self):
        self.clipboard_clear()
        self.clipboard_append(self.chat_display.get("0.0", "end"))

    def start_generation(self):
        prompt = self.input_box.get().strip()
        if not prompt: return

        self.input_box.delete(0, "end")
        self.send_btn.configure(state="disabled", fg_color="gray")
        self.status_indicator.configure(text="● Thinking...", text_color="#FFB302")
        
        # Display User Message
        self.update_chat(f"YOU: {prompt}\n", "user")
        
        # Threading prevents UI freeze
        threading.Thread(target=self.run_model, args=(prompt,)).start()

    def run_model(self, prompt):
        result = generate_text_logic(prompt, self.val_tokens, self.val_temp, self.val_topk)
        
        # Back to UI thread
        self.after(0, lambda: self.finish_generation(result))

    def finish_generation(self, result):
        self.update_chat(f"GPT: {result}\n\n", "ai")
        self.send_btn.configure(state="normal", fg_color=COLOR_ACCENT)
        self.status_indicator.configure(text="● Online", text_color="#4CC38A")

    def update_chat(self, text, sender):
        self.chat_display.configure(state="normal")
        
        # Simple separator logic
        if sender == "user":
            self.chat_display.insert("end", "┌──────────────────────────────────────────\n")
            self.chat_display.insert("end", f"│ {text}")
        else:
            self.chat_display.insert("end", f"│ {text}")
            self.chat_display.insert("end", "└──────────────────────────────────────────\n\n")
            
        self.chat_display.see("end")
        self.chat_display.configure(state="disabled")

if __name__ == "__main__":
    app = ModernGPTApp()
    app.mainloop()