# Airfoil Lab - AI-Enhanced Aerodynamic Design

**Airfoil Lab** is a modern, interactive web application for designing and analyzing airfoils. It combines real-time visualization, accurate XFOIL simulations, and AI-powered assistance to help students and engineers understand aerodynamics intuitively.

[中文设计文档 (Design Document)](./design_doc_zh.md)

## ✨ Key Features

- **Interactive Design**: Adjust NACA 4-digit parameters (Camber, Thickness, Position) with instant visual feedback.
- **Real-time Preview**: 
  - **Generative Airfoil**: See the shape change as you slide parameters.
  - **Pressure Heatmap**: Visualize suction and pressure zones instantly (estimated via Thin Airfoil Theory).
  - **Pressure Vectors**: View Cp vectors normal to the surface.
- **Accurate Simulation**: 
  - Integrated **XFOIL** solver for precise aerodynamic coefficients ($C_l, C_d, C_m, C_p$).
  - Automatic fallback to estimation algorithms if XFOIL is unavailable.
- **Data Visualization**:
  - Interactive **Polar Charts** ($C_l$ vs $\alpha$, $C_l$ vs $C_d$).
  - **$C_p$ Distribution** charts.
- **History & Comparison**: Save design iterations and compare their performance side-by-side.
- **AI Assistant**: Context-aware chat bot to answer aerodynamics questions and guide optimization.

## 🛠 Tech Stack

### Frontend
- **Next.js 14** (App Router)
- **TypeScript**
- **Tailwind CSS** + **Shadcn UI**
- **Zustand** (State Management)
- **Recharts** (Data Visualization)

### Backend
- **FastAPI** (Python)
- **SQLite** + **SQLAlchemy**
- **Pandas / NumPy** (Data Processing)
- **XFOIL** (Aerodynamic Solver)

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- Python 3.9+

### 1. Backend Setup
The backend handles simulations and database operations.

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server (runs on localhost:8000)
python -m uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Frontend Setup
The frontend provides the user interface.

```bash
cd airfoil-lab-react

# Install dependencies
npm install

# Start the development server (runs on localhost:3000)
npm run dev
```

### 3. Usage
Open [http://localhost:3000](http://localhost:3000) in your browser.
- **Default Admin Account**: 
  - Username: `admin`
  - Password: `ecnusjtu`

## 📂 Project Structure

```
├── airfoil-lab-react/      # Frontend source code
│   ├── src/app/            # Next.js pages and layouts
│   ├── src/components/     # React components (Airfoil, Charts, UI)
│   └── src/lib/            # Utility functions (Geometry generation)
├── backend.py              # Main backend application file
├── aero_data.db            # SQLite database (Simulations & History)
├── design_doc_zh.md        # Technical Design Document (Chinese)
└── requirements.txt        # Python dependencies
```

## 🤝 Contributing
1. Create a new branch: `git checkout -b feature/my-feature`
2. Commit your changes: `git commit -m "Add new feature"`
3. Push to the branch: `git push origin feature/my-feature`
4. Submit a Pull Request.

## 📄 License
MIT License.
