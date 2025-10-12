import { useState } from 'react'
import { Upload, Loader2, AlertCircle, CheckCircle2, TrendingUp } from 'lucide-react'
import Papa from 'papaparse'
import * as XLSX from 'xlsx'

const API_BASE_URL = '/api'

// Colores oficiales de ODS
const ODS_COLORS = {
  1: '#E5243B',  // Rojo - Fin de la pobreza
  3: '#4C9F38',  // Verde - Salud y bienestar
  4: '#C5192D'   // Rojo oscuro - Educación de calidad
}

const ODS_NAMES = {
  1: 'Fin de la pobreza',
  3: 'Salud y bienestar',
  4: 'Educación de calidad'
}

function RetrainingTab() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      const ext = selectedFile.name.split('.').pop().toLowerCase()
      if (ext === 'csv' || ext === 'xlsx') {
        setFile(selectedFile)
        setError(null)
      } else {
        setError('Por favor, seleccione un archivo CSV o XLSX')
        setFile(null)
      }
    }
  }

  const parseFile = async (file) => {
    return new Promise((resolve, reject) => {
      const ext = file.name.split('.').pop().toLowerCase()
      
      if (ext === 'csv') {
        Papa.parse(file, {
          header: true,
          complete: (results) => {
            try {
              const data = results.data.filter(row => row && Object.keys(row).length > 0)
              
              const textColumn = Object.keys(data[0]).find(key => 
                key.toLowerCase().includes('texto') || key.toLowerCase().includes('text')
              )
              const labelColumn = Object.keys(data[0]).find(key => 
                key.toLowerCase().includes('label') || key.toLowerCase().includes('etiqueta')
              )

              if (!textColumn || !labelColumn) {
                throw new Error('El archivo debe contener columnas "textos" y "labels"')
              }

              const texts = data.map(row => String(row[textColumn]).trim()).filter(Boolean)
              const labels = data.map(row => parseInt(row[labelColumn])).filter(val => !isNaN(val))

              if (texts.length !== labels.length) {
                throw new Error('El número de textos y labels no coincide')
              }

              resolve({ textos: texts, labels: labels })
            } catch (err) {
              reject(err)
            }
          },
          error: (error) => reject(error),
          skipEmptyLines: true
        })
      } else if (ext === 'xlsx') {
        const reader = new FileReader()
        reader.onload = (e) => {
          try {
            const data = new Uint8Array(e.target.result)
            const workbook = XLSX.read(data, { type: 'array' })
            const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
            const jsonData = XLSX.utils.sheet_to_json(firstSheet)

            if (jsonData.length === 0) {
              throw new Error('El archivo está vacío')
            }

            const textColumn = Object.keys(jsonData[0]).find(key => 
              key.toLowerCase().includes('texto') || key.toLowerCase().includes('text')
            )
            const labelColumn = Object.keys(jsonData[0]).find(key => 
              key.toLowerCase().includes('label') || key.toLowerCase().includes('etiqueta')
            )

            if (!textColumn || !labelColumn) {
              throw new Error('El archivo debe contener columnas "textos" y "labels"')
            }

            const texts = jsonData.map(row => String(row[textColumn]).trim()).filter(Boolean)
            const labels = jsonData.map(row => parseInt(row[labelColumn])).filter(val => !isNaN(val))

            if (texts.length !== labels.length) {
              throw new Error('El número de textos y labels no coincide')
            }

            resolve({ textos: texts, labels: labels })
          } catch (err) {
            reject(err)
          }
        }
        reader.onerror = (error) => reject(error)
        reader.readAsArrayBuffer(file)
      }
    })
  }

  const handleRetrain = async () => {
    if (!file) {
      setError('Por favor, seleccione un archivo')
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const { textos, labels } = await parseFile(file)

      if (textos.length === 0) {
        throw new Error('El archivo no contiene datos válidos')
      }

      const response = await fetch(`${API_BASE_URL}/retrain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ textos, labels })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Error en el reentrenamiento')
      }

      const data = await response.json()
      setResults(data)
      setSuccess(`Reentrenamiento completado con ${textos.length} ejemplos`)
    } catch (err) {
      setError(err.message || 'Error al procesar el reentrenamiento')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-start gap-3">
          <AlertCircle className="text-red-500 flex-shrink-0" size={20} />
          <p className="text-red-700">{error}</p>
        </div>
      )}

      {success && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4 flex items-start gap-3">
          <CheckCircle2 className="text-green-500 flex-shrink-0" size={20} />
          <p className="text-green-700">{success}</p>
        </div>
      )}

      <div className="bg-green-50 border border-green-200 rounded-lg p-4">
        <h4 className="font-semibold text-green-900 mb-2">Instrucciones</h4>
        <ul className="text-sm text-green-800 space-y-1 list-disc list-inside">
          <li>El archivo debe contener dos columnas: <strong>"textos"</strong> y <strong>"labels"</strong></li>
          <li>Cada fila representa un ejemplo de entrenamiento</li>
          <li>Las etiquetas válidas son: <strong>1</strong> (Fin de la pobreza), <strong>3</strong> (Salud y bienestar), <strong>4</strong> (Educación de calidad)</li>
          <li>Formatos aceptados: CSV o XLSX</li>
        </ul>
      </div>

      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Upload size={20} />
          Subir Datos de Entrenamiento
        </h3>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="flex-1">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 hover:border-green-400 transition-colors cursor-pointer">
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="text-center">
                  <Upload className="mx-auto text-gray-400 mb-3" size={40} />
                  <p className="text-gray-700 font-medium mb-1">
                    {file ? file.name : 'Seleccionar archivo de entrenamiento'}
                  </p>
                  <p className="text-sm text-gray-500">
                    CSV o XLSX con columnas "textos" y "labels"
                  </p>
                </div>
              </div>
            </label>
          </div>
          <button
            onClick={handleRetrain}
            disabled={loading || !file}
            className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-6 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={20} />
                Reentrenando modelo...
              </>
            ) : (
              'Reentrenar Modelo'
            )}
          </button>
        </div>
      </div>

      {results && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-6 flex items-center gap-2">
            <TrendingUp size={20} />
            Métricas del Modelo
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <MetricCard
              label="Precisión (Macro)"
              value={(results.precision * 100).toFixed(2)}
              color="orange"
            />
            <MetricCard
              label="Recall (Macro)"
              value={(results.recall * 100).toFixed(2)}
              color="red"
            />
            <MetricCard
              label="F1-Score (Macro)"
              value={(results.f1_score * 100).toFixed(2)}
              color="amber"
            />
          </div>

          {results.precision_per_class && results.recall_per_class && (
            <div className="mt-6">
              <h4 className="font-semibold text-gray-700 mb-3">Métricas por Clase (ODS)</h4>
              <div className="space-y-3">
                {results.precision_per_class.map((precision, idx) => {
                  const odsNumber = idx + 1
                  const odsColor = ODS_COLORS[odsNumber] || '#666'
                  const odsName = ODS_NAMES[odsNumber] || `ODS ${odsNumber}`
                  
                  return (
                    <div 
                      key={idx} 
                      className="p-4 rounded-lg text-white"
                      style={{backgroundColor: odsColor}}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <h5 className="font-bold text-lg">ODS {odsNumber}: {odsName}</h5>
                      </div>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <span className="opacity-90">Precisión:</span>
                          <span className="ml-2 font-bold">{(precision * 100).toFixed(2)}%</span>
                        </div>
                        <div>
                          <span className="opacity-90">Recall:</span>
                          <span className="ml-2 font-bold">{(results.recall_per_class[idx] * 100).toFixed(2)}%</span>
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function MetricCard({ label, value, color }) {
  const colorClasses = {
    orange: 'bg-green-50 text-green-700 border-green-200',
    red: 'bg-green-100 text-green-800 border-green-300',
    amber: 'bg-green-200 text-green-900 border-green-400'
  }

  return (
    <div className={`border rounded-lg p-4 ${colorClasses[color]}`}>
      <p className="text-sm font-medium mb-1">{label}</p>
      <p className="text-3xl font-bold">{value}%</p>
    </div>
  )
}

export default RetrainingTab
