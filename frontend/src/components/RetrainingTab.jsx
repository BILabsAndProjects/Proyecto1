import { useState, useEffect } from 'react'
import Papa from 'papaparse'
import * as XLSX from 'xlsx'
import { Upload, Loader2, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts'

const API_BASE_URL = '/api'

// Colores oficiales de ODS
const ODS_COLORS = {
  1: '#E5243B',  // Rojo - Fin de la pobreza
  3: '#4C9F38',  // Verde - Salud y bienestar
  4: '#C5192D'   // Rojo oscuro - Educación de calidad
}

const ODS_NAMES = {
  1: 'FIN DE LA POBREZA',
  3: 'SALUD Y BIENESTAR',
  4: 'EDUCACIÓN DE CALIDAD'
}

function RetrainingTab() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)
  const [success, setSuccess] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  // Cargar información del modelo al montar el componente y después de reentrenar
  const fetchModelInfo = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/model-info`)
      if (response.ok) {
        const data = await response.json()
        console.log('Model info received:', data)
        setModelInfo(data)
      }
    } catch (err) {
      console.error('Error al cargar información del modelo:', err)
    }
  }

  useEffect(() => {
    fetchModelInfo()
  }, [])

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
              
              if (data.length === 0) {
                throw new Error('El archivo está vacío')
              }
              
              const columns = Object.keys(data[0])
              
              // Intentar encontrar columnas por nombre
              let textColumn = columns.find(key => 
                key.toLowerCase().includes('texto') || key.toLowerCase().includes('text')
              )
              let labelColumn = columns.find(key => 
                key.toLowerCase().includes('label') || key.toLowerCase().includes('etiqueta')
              )

              // Si no se encuentran por nombre y hay exactamente 2 columnas, usar las 2 primeras
              if ((!textColumn || !labelColumn) && columns.length === 2) {
                textColumn = columns[0]
                labelColumn = columns[1]
              }

              if (!textColumn || !labelColumn) {
                throw new Error('El archivo debe contener 2 columnas o columnas llamadas "textos" y "labels"')
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

            const columns = Object.keys(jsonData[0])
            
            // Intentar encontrar columnas por nombre
            let textColumn = columns.find(key => 
              key.toLowerCase().includes('text')
            )
            let labelColumn = columns.find(key => 
              key.toLowerCase().includes('label') || key.toLowerCase().includes('etiqueta')
            )

            // Si no se encuentran por nombre y hay exactamente 2 columnas, usar las 2 primeras
            if ((!textColumn || !labelColumn) && columns.length === 2) {
              textColumn = columns[0]
              labelColumn = columns[1]
            }

            if (!textColumn || !labelColumn) {
              throw new Error('El archivo debe contener 2 columnas o columnas llamadas "textos" y "labels"')
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
      const timestamp = data.model_timestamp ? ` - Modelo guardado: ${data.model_timestamp}` : ''
      setSuccess(`Reentrenamiento completado con ${textos.length} ejemplos${timestamp}`)
      // Actualizar información del modelo después del reentrenamiento
      fetchModelInfo()
    } catch (err) {
      setError(err.message || 'Error al procesar el reentrenamiento')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Información del Modelo */}
      {modelInfo && (
        <div className={`rounded-lg p-4 text-center ${
          modelInfo.model_timestamp 
            ? 'bg-green-100 border-2 border-green-400' 
            : 'bg-gray-50 border border-gray-200'
        }`}>
          {modelInfo.model_timestamp ? (
            <div className="flex flex-col items-center gap-1">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-green-700">Última fecha de entrenamiento:</span>
                <span className="text-base text-green-900 font-bold px-3 py-1 rounded">
                  {(() => {
                    const timestamp = modelInfo.model_timestamp
                    const year = timestamp.substring(0, 4)
                    const month = timestamp.substring(4, 6)
                    const day = timestamp.substring(6, 8)
                    const hour = timestamp.substring(9, 11)
                    const minute = timestamp.substring(11, 13)
                    const second = timestamp.substring(13, 15)
                    return `${day}/${month}/${year} ${hour}:${minute}:${second}`
                  })()}
                </span>
              </div>
            </div>
          ) : (
            <span className="text-sm font-medium text-gray-700">
              Usando modelo original - No hay reentrenamientos previos
            </span>
          )}
        </div>
      )}

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
          <li>El archivo debe contener dos columnas con textos y etiquetas</li>
          <li>Se aceptan nombres como <strong>"textos"</strong> y <strong>"labels"</strong>, o cualquier archivo con exactamente 2 columnas</li>
          <li>Las etiquetas válidas son: <strong>1</strong> (FIN DE LA POBREZA), <strong>3</strong> (SALUD Y BIENESTAR), <strong>4</strong> (EDUCACIÓN DE CALIDAD)</li>
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

          {results.precision_per_class && results.recall_per_class && results.classes && (
            <div className="mt-6 space-y-6">
              {/* Gráfico de Precisión por Clase */}
              <div className="bg-gray-50 rounded-lg p-6">
                <h4 className="font-semibold text-gray-700 mb-4">Precisión por Clase</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={results.classes.map((cls, idx) => ({
                      name: `ODS ${cls}`,
                      precision: (results.precision_per_class[idx] * 100).toFixed(2)
                    }))}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="precision" fill="#10b981" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Gráfico de Recall por Clase */}
              <div className="bg-gray-50 rounded-lg p-6">
                <h4 className="font-semibold text-gray-700 mb-4">Recall (Cobertura) por Clase</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={results.classes.map((cls, idx) => ({
                      name: `ODS ${cls}`,
                      recall: (results.recall_per_class[idx] * 100).toFixed(2)
                    }))}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 100]} />
                    <Tooltip formatter={(value) => `${value}%`} />
                    <Bar dataKey="recall" fill="#10b981" radius={[8, 8, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Matriz de Confusión */}
              {results.confusion_matrix && (
                <div className="bg-gray-50 rounded-lg p-6">
                  <h4 className="font-semibold text-gray-700 mb-4">Matriz de Confusión</h4>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="border border-gray-300 bg-gray-100 p-3"></th>
                          {results.classes.map(cls => (
                            <th key={cls} className="border border-gray-300 bg-gray-100 p-3 font-semibold">
                              Predicho ODS {cls}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {results.confusion_matrix.map((row, rowIdx) => (
                          <tr key={rowIdx}>
                            <th className="border border-gray-300 bg-gray-100 p-3 font-semibold">
                              Real ODS {results.classes[rowIdx]}
                            </th>
                            {row.map((value, colIdx) => (
                              <td 
                                key={colIdx} 
                                className="border border-gray-300 p-3 text-center font-medium"
                                style={{
                                  backgroundColor: rowIdx === colIdx 
                                    ? 'rgba(16, 185, 129, 0.2)' 
                                    : value > 0 
                                    ? 'rgba(239, 68, 68, 0.1)' 
                                    : 'white'
                                }}
                              >
                                {value}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
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
