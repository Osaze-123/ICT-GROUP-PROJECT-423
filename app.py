import flet as ft
import numpy as np
import joblib
import tensorflow as tf
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "iot_model.keras"
SCALER_PATH = "iot_scaler.pkl"
TIME_STEPS = 30  # Same as training
DEFAULT_THRESHOLD = 0.15 # Adjust based on your training script output

class AnomalyChecker(ft.UserControl):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.load_assets()

    def load_assets(self):
        self.status = "Ready"
        self.status_color = ft.colors.GREEN
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.scaler = joblib.load(SCALER_PATH)
            else:
                self.status = "Error: Model/Scaler files missing!"
                self.status_color = ft.colors.RED
        except Exception as e:
            self.status = f"Error loading files: {str(e)}"
            self.status_color = ft.colors.RED

    def build(self):
        # Input Fields with "Normal" defaults
        self.txt_temp = ft.TextField(label="Temperature (°C)", value="25.0", width=150, text_align=ft.TextAlign.RIGHT)
        self.txt_hum = ft.TextField(label="Humidity (%)", value="50.0", width=150, text_align=ft.TextAlign.RIGHT)
        self.txt_press = ft.TextField(label="Pressure (hPa)", value="1013.0", width=150, text_align=ft.TextAlign.RIGHT)
        self.txt_soil = ft.TextField(label="Soil Moisture", value="800.0", width=150, text_align=ft.TextAlign.RIGHT)
        
        # Result Display
        self.lbl_result = ft.Text("Waiting for input...", size=20, weight=ft.FontWeight.BOLD)
        self.lbl_score = ft.Text("Anomaly Score: 0%", color=ft.colors.GREY_400)
        self.progress_bar = ft.ProgressBar(width=400, value=0, color=ft.colors.BLUE, bgcolor=ft.colors.GREY_800)
        self.lbl_loss = ft.Text("Raw Reconstruction Error: 0.0000", size=12, italic=True, color=ft.colors.GREY_500)

        # Container for the result card
        self.result_card = ft.Container(
            content=ft.Column([
                ft.Text("Analysis Result", size=14, color=ft.colors.GREY_400),
                self.lbl_result,
                ft.Divider(color=ft.colors.GREY_700),
                self.lbl_score,
                self.progress_bar,
                self.lbl_loss
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            padding=20,
            bgcolor=ft.colors.SURFACE_VARIANT,
            border_radius=10,
            visible=False # Hidden until first check
        )

        return ft.Column([
            ft.Text("Manual Sensor Check", size=24, weight=ft.FontWeight.BOLD),
            ft.Text(self.status, color=self.status_color, size=12),
            ft.Divider(),
            
            # Inputs Layout
            ft.Row([
                ft.Column([self.txt_temp, self.txt_hum]),
                ft.Column([self.txt_press, self.txt_soil]),
            ], alignment=ft.MainAxisAlignment.CENTER),
            
            ft.Container(height=20),
            
            # Action Button
            ft.ElevatedButton(
                "Analyze Readings", 
                icon=ft.icons.ANALYTICS, 
                style=ft.ButtonStyle(bgcolor=ft.colors.BLUE_600, color=ft.colors.WHITE),
                on_click=self.analyze_inputs,
                height=50,
                width=200
            ),
            
            ft.Container(height=20),
            self.result_card
        ], horizontal_alignment=ft.CrossAxisAlignment.CENTER)

    def analyze_inputs(self, e):
        if not self.model or not self.scaler:
            self.lbl_result.value = "Model not loaded!"
            self.lbl_result.color = ft.colors.RED
            self.result_card.visible = True
            self.update()
            return

        try:
            # 1. Get Values
            val_temp = float(self.txt_temp.value)
            val_hum = float(self.txt_hum.value)
            val_press = float(self.txt_press.value)
            val_soil = float(self.txt_soil.value)

            # 2. Preprocess
            # Create a single row of data
            input_vector = np.array([[val_temp, val_hum, val_press, val_soil]])
            # Scale it
            scaled_vector = self.scaler.transform(input_vector)
            
            # 3. Create Sequence
            # Since LSTM expects 30 time steps, we repeat this reading 30 times
            # Effectively checking: "Is a steady state of these values normal?"
            input_sequence = np.repeat(scaled_vector, TIME_STEPS, axis=0)
            input_batch = input_sequence.reshape(1, TIME_STEPS, 4)

            # 4. Predict
            reconstruction = self.model.predict(input_batch, verbose=0)
            
            # 5. Calculate Error (Mean Absolute Error)
            mae_loss = np.mean(np.abs(reconstruction - input_batch))
            
            # 6. Determine Status
            # Calculate a percentage score relative to threshold (100% = Threshold)
            score_percent = (mae_loss / DEFAULT_THRESHOLD)
            
            self.lbl_loss.value = f"Raw Reconstruction Error: {mae_loss:.4f}"
            self.result_card.visible = True

            if mae_loss > DEFAULT_THRESHOLD:
                self.lbl_result.value = "ANOMALY DETECTED"
                self.lbl_result.color = ft.colors.RED
                self.lbl_score.value = f"Severity: {score_percent*100:.1f}% (Above Normal Limits)"
                self.progress_bar.color = ft.colors.RED
                self.progress_bar.value = min(score_percent * 0.5, 1.0) # Cap at 1.0 for UI
            else:
                self.lbl_result.value = "STATUS NORMAL"
                self.lbl_result.color = ft.colors.GREEN
                self.lbl_score.value = f"Deviation: {score_percent*100:.1f}% (Within Normal Limits)"
                self.progress_bar.color = ft.colors.GREEN
                self.progress_bar.value = score_percent

            self.update()

        except ValueError:
            self.lbl_result.value = "Invalid Input: Please enter numbers only."
            self.lbl_result.color = ft.colors.RED
            self.result_card.visible = True
            self.update()

def main(page: ft.Page):
    page.title = "Simple IoT Diagnostic Tool"
    page.theme_mode = ft.ThemeMode.DARK
    page.window_width = 500
    page.window_height = 700
    page.padding = 30
    page.vertical_alignment = ft.MainAxisAlignment.CENTER
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    page.add(AnomalyChecker())

if __name__ == "__main__":
    ft.app(target=main)