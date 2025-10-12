import { useState } from 'react'
import { Upload, FileText, Download, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react'
import Papa from 'papaparse'
import * as XLSX from 'xlsx'

const API_BASE_URL = '/api'

// Colores oficiales de ODS
const ODS_COLORS = {
  1: '#E5243B',  // Rojo - Fin de la pobreza
  3: '#4C9F38',  // Verde - Salud y bienestar
  4: '#C5192D'   // Rojo oscuro - Educación de calidad
}

function PredictionTab() {
  const [inputText, setInputText] = useState('')
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
          complete: (results) => {
            // Extrae la primera columna de todas las filas (excepto header si existe)
            const texts = results.data
              .filter(row => row && row[0] && row[0].trim())
              .map(row => row[0].trim())
            resolve(texts)
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
            const jsonData = XLSX.utils.sheet_to_json(firstSheet, { header: 1 })
            
            // Extrae la primera columna
            const texts = jsonData
              .filter(row => row && row[0] && String(row[0]).trim())
              .map(row => String(row[0]).trim())
            resolve(texts)
          } catch (err) {
            reject(err)
          }
        }
        reader.onerror = (error) => reject(error)
        reader.readAsArrayBuffer(file)
      }
    })
  }

  const handleTextPredict = async () => {
    if (!inputText.trim()) {
      setError('Por favor, ingrese un texto para predecir')
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ textos: [inputText.trim()] })
      })

      if (!response.ok) throw new Error('Error en la predicción')

      const data = await response.json()
      setResults(data)
      setSuccess('Predicción completada exitosamente')
    } catch (err) {
      setError(err.message || 'Error al realizar la predicción')
    } finally {
      setLoading(false)
    }
  }

  const handleFilePredict = async () => {
    if (!file) {
      setError('Por favor, seleccione un archivo')
      return
    }

    setLoading(true)
    setError(null)
    setSuccess(null)

    try {
      const texts = await parseFile(file)
      
      if (texts.length === 0) {
        throw new Error('El archivo está vacío o no contiene datos válidos')
      }

      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ textos: texts })
      })

      if (!response.ok) throw new Error('Error en la predicción')

      const data = await response.json()
      setResults(data)
      setSuccess(`Predicción completada: ${data.length} textos procesados`)
    } catch (err) {
      setError(err.message || 'Error al procesar el archivo')
    } finally {
      setLoading(false)
    }
  }

  const downloadResults = () => {
    if (!results) return

    // Crear CSV con resultados
    const csvRows = [
      ['Índice', 'Predicción', 'Probabilidades'].join(',')
    ]

    results.forEach((result, idx) => {
      const probsStr = Object.entries(result.probabilities)
        .map(([cls, prob]) => `${cls}:${prob.toFixed(4)}`)
        .join(' | ')
      csvRows.push([idx + 1, result.prediction, `"${probsStr}"`].join(','))
    })

    const csvContent = csvRows.join('\n')
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const link = document.createElement('a')
    link.href = URL.createObjectURL(blob)
    link.download = 'predicciones.csv'
    link.click()
  }

  return (
    <div className="space-y-6">
      {/* Alerts */}
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

      {/* Text Input Section */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <FileText size={20} />
          Predicción Manual
        </h3>
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Ingrese el texto a analizar..."
          className="w-full p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-orange-500 focus:border-transparent resize-none"
          rows="4"
        />
        <button
          onClick={handleTextPredict}
          disabled={loading}
          className="mt-4 bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" size={18} />
              Procesando...
            </>
          ) : (
            'Predecir'
          )}
        </button>
      </div>

      {/* File Upload Section */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
          <Upload size={20} />
          Subir Archivo
        </h3>
        <div className="space-y-4">
          <div className="flex items-center gap-4">
            <label className="flex-1">
              <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 hover:border-orange-400 transition-colors cursor-pointer">
                <input
                  type="file"
                  accept=".csv,.xlsx"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="text-center">
                  <Upload className="mx-auto text-gray-400 mb-2" size={32} />
                  <p className="text-gray-600">
                    {file ? file.name : 'Seleccionar archivo CSV o XLSX'}
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    El archivo debe contener una columna con los textos
                  </p>
                </div>
              </div>
            </label>
          </div>
          <button
            onClick={handleFilePredict}
            disabled={loading || !file}
            className="bg-orange-500 hover:bg-orange-600 text-white font-medium py-2 px-6 rounded-lg transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="animate-spin" size={18} />
                Procesando...
              </>
            ) : (
              'Predecir desde Archivo'
            )}
          </button>
        </div>
      </div>

      {/* Results Section */}
      {results && results.length > 0 && (
        <div className="bg-white border border-gray-200 rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">
              Resultados ({results.length} predicciones)
            </h3>
            <button
              onClick={downloadResults}
              className="bg-green-500 hover:bg-green-600 text-white font-medium py-2 px-4 rounded-lg transition-colors flex items-center gap-2"
            >
              <Download size={18} />
              Descargar CSV
            </button>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">#</th>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">ODS Predicho</th>
                  <th className="px-4 py-2 text-left text-sm font-semibold text-gray-700">Probabilidades</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.map((result, idx) => (
                  <tr key={idx} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm text-gray-600">{idx + 1}</td>
                    <td className="px-4 py-3">
                      <span 
                        className="inline-block px-3 py-1 rounded-full text-sm font-medium text-white"
                        style={{backgroundColor: ODS_COLORS[result.prediction] || '#666'}}
                      >
                        ODS {result.prediction}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="space-y-1">
                        {[1, 3, 4].map((odsNum) => {
                          const prob = result.probabilities[odsNum]
                          return prob !== undefined ? (
                            <div key={odsNum} className="flex items-center gap-2">
                              <span className="text-xs text-gray-600 w-12">ODS {odsNum}:</span>
                              <div className="flex-1 bg-gray-200 rounded-full h-2 max-w-[100px]">
                                <div
                                  className="h-2 rounded-full"
                                  style={{ 
                                    width: `${prob * 100}%`,
                                    backgroundColor: '#f97316'
                                  }}
                                />
                              </div>
                              <span className="text-xs text-gray-600 w-12">
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                          ) : null
                        })}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default PredictionTab
