import { useState } from 'react'
import './App.css'

function App() {
  const [uploadedFiles, setUploadedFiles] = useState([])
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files)
    setUploadedFiles(prev => [...prev, ...files])
  }

  const handleQuerySubmit = async (e) => {
    e.preventDefault()
    if (!query.trim()) return
    
    setIsProcessing(true)
    // Simulate processing time
    setTimeout(() => {
      setResults({
        answer: "Based on the uploaded documents, the relevant clauses indicate...",
        sourceClause: "Section 4.2.1 of the contract states...",
        confidence: 95,
        documentSource: "Contract_Agreement_2024.pdf"
      })
      setIsProcessing(false)
    }, 2000)
  }

  const removeFile = (index) => {
    setUploadedFiles(prev => prev.filter((_, i) => i !== index))
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">ClauseNaut</h1>
        <p className="app-subtitle">AI-Powered Document Intelligence & Clause Discovery Platform</p>
      </header>

      <main className="main-content">
        {/* Document Upload Section */}
        <section className="upload-section glass-card">
          <h2>Upload Documents</h2>
          <div className="upload-area">
            <input
              type="file"
              multiple
              accept=".pdf,.doc,.docx,.eml"
              onChange={handleFileUpload}
              className="file-input"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="upload-label">
              <div className="upload-icon">🚀</div>
              <span>Drag & drop your documents here</span>
              <small>Supports PDF, Word documents, and emails</small>
            </label>
          </div>
          
          {uploadedFiles.length > 0 && (
            <div className="uploaded-files">
              <h3>Processing Queue ({uploadedFiles.length} files)</h3>
              {uploadedFiles.map((file, index) => (
                <div key={index} className="file-item">
                  <span className="file-name">{file.name}</span>
                  <button onClick={() => removeFile(index)} className="remove-btn">×</button>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Query Section */}
        <section className="query-section glass-card">
          <h2>Query Engine</h2>
          <form onSubmit={handleQuerySubmit} className="query-form">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Ask anything about your documents... Our AI understands natural language and will find relevant clauses even from vague queries."
              className="query-input"
              rows="4"
            />
            <button 
              type="submit" 
              className={`submit-btn ${isProcessing ? 'processing' : ''}`}
              disabled={isProcessing || !query.trim() || uploadedFiles.length === 0}
            >
              {isProcessing ? '🔄 Analyzing...' : '🎯 Search Clauses'}
            </button>
          </form>
        </section>

        {/* Results Section */}
        {results && (
          <section className="results-section glass-card">
            <h2>Analysis Results</h2>
            <div className="result-card">
              <div className="result-header">
                <span className="confidence-badge">
                  🎯 {results.confidence}% Match
                </span>
                <span className="source-doc">📄 {results.documentSource}</span>
              </div>
              
              <div className="result-content">
                <h3>📋 Executive Summary</h3>
                <p className="answer-text">{results.answer}</p>
                
                <h3>📖 Referenced Clause</h3>
                <div className="source-clause">
                  <p>"{results.sourceClause}"</p>
                </div>
                
                <div className="result-actions">
                  <button className="action-btn primary">📊 Export Analysis</button>
                  <button className="action-btn secondary">🔍 View Document</button>
                </div>
              </div>
            </div>
          </section>
        )}

        {/* Features Section */}
        <section className="features-section glass-card">
          <h2>Platform Capabilities</h2>
          <div className="features-grid">
            <div className="feature-card">
              <div className="feature-icon">🧠</div>
              <h3>Natural Language Processing</h3>
              <p>Query documents using everyday language - no technical syntax required</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">⚡</div>
              <h3>Multi-Format Support</h3>
              <p>Process PDFs, Word documents, and email files seamlessly</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🎯</div>
              <h3>Explainable AI</h3>
              <p>Every result includes source references and confidence metrics</p>
            </div>
            <div className="feature-card">
              <div className="feature-icon">🔒</div>
              <h3>Enterprise Ready</h3>
              <p>Built for claims processing, audits, and compliance workflows</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
