import { useState } from 'react'
import PredictionTab from './components/PredictionTab'
import RetrainingTab from './components/RetrainingTab'
import { Brain, RefreshCw } from 'lucide-react'

function App() {
  const [activeTab, setActiveTab] = useState('predict')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Sistema de Predicción de ODS
          </h1>
          <p className="text-gray-600 mb-4">
            Predicción y reentrenamiento de modelos de análisis de textos
          </p>
          <div className="bg-white rounded-lg shadow p-4 max-w-4xl mx-auto mt-4">
            <h3 className="font-semibold text-gray-800 mb-4 text-center">Objetivos de Desarrollo Sostenible (ODS) disponibles:</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex flex-col items-center">
                <img 
                  src="/ods1.jpg" 
                  alt="ODS 1 - Fin de la pobreza" 
                  className="w-full rounded-lg shadow-md hover:shadow-lg transition-shadow"
                />
                <p className="text-center text-sm text-gray-700 mt-2 font-medium">Fin de la pobreza</p>
              </div>
              <div className="flex flex-col items-center">
                <img 
                  src="/ods3.jpg" 
                  alt="ODS 3 - Salud y bienestar" 
                  className="w-full rounded-lg shadow-md hover:shadow-lg transition-shadow"
                />
                <p className="text-center text-sm text-gray-700 mt-2 font-medium">Salud y bienestar</p>
              </div>
              <div className="flex flex-col items-center">
                <img 
                  src="/ods4.jpg" 
                  alt="ODS 4 - Educación de calidad" 
                  className="w-full rounded-lg shadow-md hover:shadow-lg transition-shadow"
                />
                <p className="text-center text-sm text-gray-700 mt-2 font-medium">Educación de calidad</p>
              </div>
            </div>
          </div>
        </header>

        {/* Tabs */}
        <div className="max-w-6xl mx-auto">
          <div className="bg-white rounded-lg shadow-lg overflow-hidden">
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200">
              <button
                onClick={() => setActiveTab('predict')}
                className={`flex-1 px-6 py-4 text-center font-medium transition-colors duration-200 flex items-center justify-center gap-2 ${
                  activeTab === 'predict'
                    ? 'bg-orange-500 text-white border-b-2 border-orange-600'
                    : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                }`}
              >
                <Brain size={20} />
                Predicción
              </button>
              <button
                onClick={() => setActiveTab('retrain')}
                className={`flex-1 px-6 py-4 text-center font-medium transition-colors duration-200 flex items-center justify-center gap-2 ${
                  activeTab === 'retrain'
                    ? 'bg-green-600 text-white border-b-2 border-green-700'
                    : 'bg-gray-50 text-gray-700 hover:bg-gray-100'
                }`}
              >
                <RefreshCw size={20} />
                Reentrenamiento
              </button>
            </div>

            {/* Tab Content */}
            <div className="p-6">
              {activeTab === 'predict' ? <PredictionTab /> : <RetrainingTab />}
            </div>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-8 text-gray-600 text-sm">
          <p>Proyecto 1 - Análisis de Textos ODS</p>
        </footer>
      </div>
    </div>
  )
}

export default App
