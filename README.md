# 🌾 RiceTec Sales & Opportunity Intelligence Chatbot

A conversational intelligence assistant and performance testing dashboard for RiceTec sales data (`RiceTec Sales  Opportunity Navpreet.xlsx`).

---

## Option 1: Direct Link (Password Protected)

Your teammate can access the chatbot directly in their browser from any device:

🔗 **Primary Direct Link**: **[https://6896aa86e9ac60.lhr.life](https://6896aa86e9ac60.lhr.life)**  
🔑 **Access Password**: `ricetec`

*(Alternative persistent link: [https://easy-dolls-boil.loca.lt](https://easy-dolls-boil.loca.lt) • Gateway IP: `122.183.40.95`)*  
*(If on the same office Wi-Fi / LAN: [http://192.168.1.2:8501](http://192.168.1.2:8501))*

- **Encrypted Transmission**: Uses end-to-end TLS encryption (`https://`).
- **Gated Access**: The application and all data remain locked behind the password screen until `ricetec` is entered.
- **In-Memory Security**: Raw files are never stored or uploaded externally; queries are computed dynamically on your machine.
*(If on the same office Wi-Fi / LAN, they can also use [http://192.168.1.2:8501](http://192.168.1.2:8501) with the same password).*

---

## Option 2: Run on Another Local Computer (1-Click)

To run this application on another computer without deploying to the cloud:

1. Copy or zip this folder (`Chatbot_USSales`) onto the other computer.
2. Double-click **`run_app.bat`**.
   - It will automatically verify Python, install the few required lightweight dependencies (`pandas`, `openpyxl`, `streamlit`, `altair`), and launch the interface in the default web browser.

### Manual Command Line Launch:
```powershell
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## How to Ask Questions / Type Queries

### 1. In the Web Interface (`http://localhost:8501` or `http://192.168.1.2:8501`)
- **Top Input Bar**: Type in the text box labeled *"Type your question here..."* right above the chat messages and hit **Enter** or click **"🚀 Ask"**.
- **Bottom Chat Bar**: Or type directly into the chat input bar at the bottom of the browser.
- **Quick Buttons**: Click any of the prompt suggestion buttons (e.g., *Top 10 customers*, *Product mix*).

### 2. In the Terminal / Command Prompt (`chat.py`)
If someone prefers typing queries directly in their terminal without a browser:
```powershell
python chat.py
```
This opens an interactive command-line session where you can type queries like:
- `Tell me about DONNY DELINE`
- `Top 10 customers by 2025 acres`
- `What is the 2025 product mix?`
- `Super Loyalty customers`
- `Summarize Region R01`
