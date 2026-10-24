import React, { useState } from 'react'
import './App.css'
import ASLTranslator from './components/ASLTranslator'
import SpeechToASL from './components/SpeechToASL'

function App() {
  const [activeTab, setActiveTab] = useState('webcam')

  return (
    <div className="app-container">
      <nav className="top-nav">
        <button 
          className={activeTab === 'webcam' ? 'active' : ''} 
          onClick={() => setActiveTab('webcam')}
        >
          Sign to Text (Webcam)
        </button>
        <button 
          className={activeTab === 'speech' ? 'active' : ''} 
          onClick={() => setActiveTab('speech')}
        >
          Speech to Sign (Avatar)
        </button>
      </nav>

      {activeTab === 'webcam' ? <ASLTranslator /> : <SpeechToASL />}
    </div>
  )
}

export default App
