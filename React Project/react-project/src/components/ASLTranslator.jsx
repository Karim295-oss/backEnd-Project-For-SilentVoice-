import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import './ASLTranslator.css';

const ASLTranslator = () => {
    const webcamRef = useRef(null);
    const wsRef = useRef(null);
    const activeModelRef = useRef('current');
    const requestedModelRef = useRef('current');
    const [recording, setRecording] = useState(false);
    const [sentence, setSentence] = useState([]);
    const [topPredictions, setTopPredictions] = useState([]);
    const [selectedModel, setSelectedModel] = useState('current');
    const [status, setStatus] = useState("Connected. Press SPACE to Record.");

    useEffect(() => {
        // Connect to FastAPI WebSocket via Vite proxy
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        wsRef.current = new WebSocket(`${wsProtocol}//${window.location.host}/ws`);
        
        wsRef.current.onopen = () => {
            console.log("WebSocket Connected");
            setStatus("Ready. Press SPACE to Record.");
            wsRef.current.send(JSON.stringify({
                action: "select_model",
                model: requestedModelRef.current,
            }));
        };
        
        wsRef.current.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'model_selected') {
                activeModelRef.current = data.model;
                requestedModelRef.current = data.model;
                setSelectedModel(data.model);
                setStatus(`${data.message}. Press SPACE to Record.`);
            } else if (data.type === 'model_error') {
                requestedModelRef.current = activeModelRef.current;
                setSelectedModel(activeModelRef.current);
                setStatus(data.message);
            } else if (data.type === 'prediction') {
                if (data.word) {
                    setSentence(prev => {
                        const newSentence = [...prev, data.word];
                        // Keep last 6 words like the python script
                        return newSentence.length > 6 ? newSentence.slice(newSentence.length - 6) : newSentence;
                    });
                }
                setTopPredictions(data.top_predictions);
                setStatus("Ready. Press SPACE to Record.");
            } else if (data.type === 'status') {
                setStatus(data.message);
            }
        };

        wsRef.current.onclose = () => {
            setStatus("Server disconnected. Is backend.py running?");
        };

        return () => {
            if (wsRef.current) wsRef.current.close();
        };
    }, []);

    const captureFrame = useCallback(() => {
        if (recording && webcamRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
            // Get base64 jpeg from webcam, downscaled — MediaPipe only needs landmarks,
            // and 640x360 is ~4x cheaper to encode/send/decode than the 720p display feed
            const imageSrc = webcamRef.current.getScreenshot({ width: 640, height: 360 });
            if (imageSrc) {
                wsRef.current.send(JSON.stringify({ action: "frame", data: imageSrc }));
            }
        }
    }, [recording]);

    useEffect(() => {
        let interval;
        if (recording) {
            // Capture at ~30 FPS
            interval = setInterval(captureFrame, 33);
        }
        return () => clearInterval(interval);
    }, [recording, captureFrame]);

    useEffect(() => {
        const handleKeyDown = (e) => {
            if (e.code === 'Space' && !e.repeat) {
                e.preventDefault();
                setRecording(prev => {
                    const nextState = !prev;
                    if (nextState) {
                        // Start recording
                        wsRef.current?.send(JSON.stringify({ action: "start_record" }));
                        setTopPredictions([]);
                    } else {
                        // Stop recording and process
                        wsRef.current?.send(JSON.stringify({ action: "stop_record" }));
                    }
                    return nextState;
                });
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, []);

    const clearSentence = () => {
        setSentence([]);
    };

    const handleModelChange = (event) => {
        const nextModel = event.target.value;
        requestedModelRef.current = nextModel;
        setSelectedModel(nextModel);
        setTopPredictions([]);
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ action: "select_model", model: nextModel }));
        }
    };

    return (
        <div className="asl-container">
            <header className="asl-header">
                <h1>SILENTVOICE</h1>
                <p>AI ASL Recognition Engine</p>
                <label className="model-selector">
                    Recognition model
                    <select
                        aria-label="Recognition model"
                        disabled={recording}
                        onChange={handleModelChange}
                        value={selectedModel}
                    >
                        <option value="current">Current model (50 signs)</option>
                        <option value="asl100">ASL-100 TGCN (100 signs)</option>
                    </select>
                </label>
            </header>
            
            <main className="asl-main">
                <div className="webcam-wrapper">
                    <Webcam
                        ref={webcamRef}
                        audio={false}
                        screenshotFormat="image/jpeg"
                        screenshotQuality={0.8}
                        videoConstraints={{ width: 1280, height: 720, facingMode: "user" }}
                        className={`webcam-feed ${recording ? 'recording' : ''}`}
                    />
                    
                    <div className="status-overlay">
                        <span className={`status-indicator ${recording ? 'pulse-red' : ''}`}></span>
                        {status}
                    </div>

                    <div className="predictions-overlay">
                        {topPredictions.map((pred, idx) => (
                            <div key={idx} className={`prediction-badge ${idx === 0 ? 'top-pred' : ''}`}>
                                {pred}
                            </div>
                        ))}
                    </div>
                </div>
                
                <div className="sentence-board">
                    <div className="sentence-display">
                        {sentence.length > 0 ? sentence.join(' ') : 'Your translated signs will appear here...'}
                    </div>
                    <button className="clear-btn" onClick={clearSentence}>Clear</button>
                </div>
            </main>
        </div>
    );
};

export default ASLTranslator;
