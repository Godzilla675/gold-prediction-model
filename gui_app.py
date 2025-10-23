#!/usr/bin/env python3
"""
Gold Price Prediction Model - GUI Application
A user-friendly interface for predicting gold prices using trained ML models.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import sys
import os
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import GoldPricePredictor
from data_fetcher import GoldDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GoldPredictionGUI:
    """GUI Application for Gold Price Prediction."""
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("Gold Price Prediction Model")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set theme colors
        self.bg_color = "#f0f0f0"
        self.header_color = "#2c3e50"
        self.accent_color = "#3498db"
        self.success_color = "#27ae60"
        self.warning_color = "#e74c3c"
        
        self.root.configure(bg=self.bg_color)
        
        # Initialize predictor
        self.predictor = None
        self.predictions = {}
        self.current_price = None
        
        # Create UI components
        self.create_widgets()
        
        # Try to load models on startup
        self.load_models_async()
        
    def create_widgets(self):
        """Create all GUI widgets."""
        
        # Header
        header_frame = tk.Frame(self.root, bg=self.header_color, height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        title_label = tk.Label(
            header_frame,
            text="🏆 Gold Price Prediction Model",
            font=("Arial", 24, "bold"),
            bg=self.header_color,
            fg="white"
        )
        title_label.pack(pady=20)
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_color)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left panel - Current Information
        left_frame = tk.LabelFrame(
            main_frame,
            text="Current Gold Price",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            fg=self.header_color,
            padx=10,
            pady=10
        )
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 10))
        
        self.current_price_label = tk.Label(
            left_frame,
            text="$---.--",
            font=("Arial", 32, "bold"),
            bg=self.bg_color,
            fg=self.header_color
        )
        self.current_price_label.pack(pady=20)
        
        self.date_label = tk.Label(
            left_frame,
            text="Loading...",
            font=("Arial", 10),
            bg=self.bg_color,
            fg="#7f8c8d"
        )
        self.date_label.pack()
        
        # Right panel - Actions
        right_frame = tk.LabelFrame(
            main_frame,
            text="Actions",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            fg=self.header_color,
            padx=10,
            pady=10
        )
        right_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 10))
        
        # Load models button
        self.load_button = tk.Button(
            right_frame,
            text="🔄 Load Models",
            font=("Arial", 11, "bold"),
            bg=self.accent_color,
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            cursor="hand2",
            command=self.load_models_async,
            height=2,
            relief=tk.FLAT
        )
        self.load_button.pack(fill=tk.X, pady=5)
        
        # Predict button
        self.predict_button = tk.Button(
            right_frame,
            text="📈 Predict Prices",
            font=("Arial", 11, "bold"),
            bg=self.success_color,
            fg="white",
            activebackground="#229954",
            activeforeground="white",
            cursor="hand2",
            command=self.make_predictions_async,
            height=2,
            relief=tk.FLAT,
            state=tk.DISABLED
        )
        self.predict_button.pack(fill=tk.X, pady=5)
        
        # Refresh data button
        self.refresh_button = tk.Button(
            right_frame,
            text="🔃 Refresh Data",
            font=("Arial", 11, "bold"),
            bg=self.accent_color,
            fg="white",
            activebackground="#2980b9",
            activeforeground="white",
            cursor="hand2",
            command=self.refresh_data_async,
            height=2,
            relief=tk.FLAT
        )
        self.refresh_button.pack(fill=tk.X, pady=5)
        
        # Status label
        self.status_label = tk.Label(
            right_frame,
            text="Ready",
            font=("Arial", 9),
            bg=self.bg_color,
            fg="#7f8c8d",
            wraplength=200
        )
        self.status_label.pack(pady=10)
        
        # Predictions panel
        pred_frame = tk.LabelFrame(
            main_frame,
            text="Model Predictions",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            fg=self.header_color,
            padx=10,
            pady=10
        )
        pred_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(10, 10))
        
        # Create treeview for predictions
        columns = ("Model", "Predicted Price", "Change ($)", "Change (%)")
        self.pred_tree = ttk.Treeview(pred_frame, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.pred_tree.heading(col, text=col)
            self.pred_tree.column(col, width=150, anchor=tk.CENTER)
        
        self.pred_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(pred_frame, orient=tk.VERTICAL, command=self.pred_tree.yview)
        self.pred_tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Log panel
        log_frame = tk.LabelFrame(
            main_frame,
            text="Activity Log",
            font=("Arial", 12, "bold"),
            bg=self.bg_color,
            fg=self.header_color,
            padx=10,
            pady=10
        )
        log_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            font=("Courier", 9),
            bg="white",
            fg="black",
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=2)
        main_frame.grid_rowconfigure(2, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Redirect logging to text widget
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to display in the text widget."""
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                def append():
                    self.text_widget.configure(state='normal')
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.configure(state='disabled')
                    self.text_widget.see(tk.END)
                self.text_widget.after(0, append)
        
        text_handler = TextHandler(self.log_text)
        text_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(text_handler)
        
    def log(self, message, level="info"):
        """Log a message to the activity log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        
        if level == "info":
            logger.info(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
            
    def update_status(self, message, color=None):
        """Update the status label."""
        self.status_label.config(text=message)
        if color:
            self.status_label.config(fg=color)
        self.root.update_idletasks()
        
    def load_models_async(self):
        """Load models in a separate thread."""
        def load():
            try:
                self.update_status("Loading models...", self.accent_color)
                self.load_button.config(state=tk.DISABLED)
                self.predict_button.config(state=tk.DISABLED)
                
                self.log("Initializing predictor...")
                self.predictor = GoldPricePredictor()
                
                self.log("Loading trained models...")
                self.predictor.load_models()
                
                self.log("Loading scalers...")
                self.predictor.load_scalers()
                
                self.log(f"Successfully loaded {len(self.predictor.models)} models")
                
                self.update_status("Models loaded successfully", self.success_color)
                self.predict_button.config(state=tk.NORMAL)
                self.load_button.config(state=tk.NORMAL)
                
                # Also fetch current price
                self.refresh_data()
                
            except Exception as e:
                self.log(f"Error loading models: {str(e)}", "error")
                self.update_status("Error loading models", self.warning_color)
                messagebox.showerror("Error", f"Failed to load models:\n{str(e)}\n\nPlease train models first using trainer.py")
                self.load_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
        
    def refresh_data_async(self):
        """Refresh current gold price data in a separate thread."""
        def refresh():
            self.refresh_data()
        
        thread = threading.Thread(target=refresh, daemon=True)
        thread.start()
        
    def refresh_data(self):
        """Fetch and display current gold price."""
        try:
            self.update_status("Fetching current data...", self.accent_color)
            self.log("Fetching latest gold price data...")
            
            fetcher = GoldDataFetcher()
            data = fetcher.fetch_data()
            
            if data is not None and len(data) > 0:
                self.current_price = data['Close'].iloc[-1]
                current_date = data.index[-1].strftime('%Y-%m-%d')
                
                self.current_price_label.config(text=f"${self.current_price:.2f}")
                self.date_label.config(text=f"As of {current_date}")
                
                self.log(f"Current gold price: ${self.current_price:.2f}")
                self.update_status("Data refreshed", self.success_color)
            else:
                self.log("No data available", "warning")
                self.update_status("No data available", self.warning_color)
                
        except Exception as e:
            self.log(f"Error fetching data: {str(e)}", "error")
            self.update_status("Error fetching data", self.warning_color)
            
    def make_predictions_async(self):
        """Make predictions in a separate thread."""
        def predict():
            try:
                self.update_status("Making predictions...", self.accent_color)
                self.predict_button.config(state=tk.DISABLED)
                
                # Clear existing predictions
                for item in self.pred_tree.get_children():
                    self.pred_tree.delete(item)
                
                self.log("Starting predictions with all models...")
                
                # Make predictions
                results = self.predictor.predict_all_models()
                self.predictions = results
                
                # Display predictions in treeview
                for model_name, result in results.items():
                    predicted_price = result['predicted_price']
                    change = result['price_change']
                    change_pct = result['price_change_pct']
                    
                    # Format change with + or - sign
                    change_str = f"${change:+.2f}"
                    change_pct_str = f"{change_pct:+.2f}%"
                    
                    self.pred_tree.insert("", tk.END, values=(
                        model_name,
                        f"${predicted_price:.2f}",
                        change_str,
                        change_pct_str
                    ))
                    
                    self.log(f"{model_name}: ${predicted_price:.2f} ({change_pct_str})")
                
                # Add ensemble prediction if multiple models
                if len(results) > 1:
                    avg_prediction = sum([r['predicted_price'] for r in results.values()]) / len(results)
                    current_price = list(results.values())[0]['current_price']
                    change = avg_prediction - current_price
                    change_pct = (change / current_price) * 100
                    
                    self.pred_tree.insert("", tk.END, values=(
                        "Ensemble (Avg)",
                        f"${avg_prediction:.2f}",
                        f"${change:+.2f}",
                        f"{change_pct:+.2f}%"
                    ), tags=('ensemble',))
                    
                    # Highlight ensemble row
                    self.pred_tree.tag_configure('ensemble', background='#e8f4f8', font=('Arial', 9, 'bold'))
                    
                    self.log(f"Ensemble prediction: ${avg_prediction:.2f} ({change_pct:+.2f}%)")
                
                self.log("Predictions completed successfully")
                self.update_status("Predictions completed", self.success_color)
                self.predict_button.config(state=tk.NORMAL)
                
            except Exception as e:
                self.log(f"Error making predictions: {str(e)}", "error")
                self.update_status("Error making predictions", self.warning_color)
                messagebox.showerror("Error", f"Failed to make predictions:\n{str(e)}")
                self.predict_button.config(state=tk.NORMAL)
        
        thread = threading.Thread(target=predict, daemon=True)
        thread.start()


def main():
    """Main entry point for the GUI application."""
    try:
        root = tk.Tk()
        app = GoldPredictionGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
