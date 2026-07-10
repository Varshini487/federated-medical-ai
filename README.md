# 🏥 Federated Medical AI

A **federated learning system** that trains a disease diagnosis model collaboratively across multiple hospitals *without* any hospital sharing raw patient data—addressing privacy regulations (HIPAA, GDPR) while improving model quality through diverse data.

## 🧠 How Federated Learning Works

### Traditional ML (Centralized)
```
Hospital A → Upload patient data → Central Server → Train model
Hospital B → Upload patient data → Central Server → Train model
Hospital C → Upload patient data → Central Server → Train model
Risk: Data breach, HIPAA violations, slow adoption
```

### Federated Learning (Decentralized)
```
Hospital A: Train model locally on its patients → Send only weights
Hospital B: Train model locally on its patients → Send only weights
Hospital C: Train model locally on its patients → Send only weights
        ↓
Central Server: Average the weights
        ↓
Broadcast updated model back to hospitals
Risk: ZERO raw data exposure, fully HIPAA compliant
```

## 🔄 The Loop
1. **Initialization:** Central server creates a baseline model and sends it to all hospitals
2. **Local Training:** Each hospital trains on its own patients for N epochs (data never leaves the hospital)
3. **Weight Upload:** Hospital sends *only* the updated weights (no patient data)
4. **Aggregation:** Central server averages weights from all hospitals (FedAvg)
5. **Distribution:** New global model sent back to all hospitals
6. **Repeat:** Cycle continues for multiple rounds until convergence

## 🛠️ Tech Stack
- Python, TensorFlow/PyTorch
- Flower framework (FL orchestration)
- FastAPI — inter-hospital communication
- SQLite — patient data (local only)
- Streamlit — dashboard

## 🚀 Getting Started
```bash
git clone https://github.com/Varshini487/federated-medical-ai
cd federated-medical-ai
pip install -r requirements.txt
# Start central server
python server.py
# In separate terminals, start client hospitals
python client.py --hospital A
python client.py --hospital B
```

## 💡 Interview Talking Points

**1. Privacy-preserving collaboration is a superpower.** Hospitals compete and can't share patient data (legal + competitive). But Federated Learning lets them train together on a shared diagnosis model without any hospital exposing their data. This is huge for adoption—hospitals get a better model (trained on 3x more diverse data) without legal risk. It's why major healthcare systems (Mayo, Cleveland Clinic) are adopting FL now.

**2. Model quality improves *because* data stays local.** A centralized model trained on 10,000 patients from one hospital may overfit to that hospital's population (age, genetics, equipment). FL trains on diverse cohorts across 3+ hospitals, so the final model generalizes better to new patients. You can prove this with experiments showing FL model accuracy on held-out test sets from hospital D (unseen during training).

**3. Addresses regulatory compliance by design.** Centralized data = HIPAA violations, data breach risk, slow regulatory approval. FL means zero raw PHI leaves the hospital, so compliance is baked in. This is why enterprises care—it's not just better accuracy, it's also safe from a legal perspective. Regulators increasingly *prefer* FL over centralized approaches.

