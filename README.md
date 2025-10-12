# Sistema de Predicción de ODS

Sistema web para predicción y reentrenamiento de modelos de clasificación de textos relacionados con Objetivos de Desarrollo Sostenible (ODS).

## 🎯 ODS Disponibles

- **ODS 1**: FIN DE LA POBREZA - Poner fin a la pobreza en todas sus formas en todo el mundo
- **ODS 3**: SALUD Y BIENESTAR - Garantizar una vida sana y promover el bienestar para todos en todas las edades
- **ODS 4**: EDUCACIÓN DE CALIDAD - Garantizar una educación inclusiva, equitativa y de calidad y promover oportunidades de aprendizaje durante toda la vida para todos

## Inicio Rápido con Docker

```bash
# Construir e iniciar
docker-compose up -d --build

# Acceder a la aplicación
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# Docs: http://localhost:8000/docs

# Detener
docker-compose down
```

## Estructura

```
├── fastAPI/          # Backend (FastAPI + scikit-learn)
├── frontend/         # Frontend (React + Vite + Tailwind)
└── docker-compose.yml

## Uso Sin Docker (Desarrollo Local)

**Terminal 1 - Backend:**
```powershell
cd fastAPI
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```powershell
cd frontend
npm install

# Configurar variable de entorno para desarrollo local
$env:VITE_API_URL="http://localhost:8000"

npm run dev
```

Luego abrir: http://localhost:3000

## Más Información

Accede a la [wiki](https://github.com/BILabsAndProjects/Proyecto1/wiki)!