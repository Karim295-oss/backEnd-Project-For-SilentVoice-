import React, { useRef, useState, useEffect } from 'react';
import './SpeechToASL.css';

const POSE_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,7],     
    [0,4],[4,5],[5,6],[6,8],     
    [9,10],                       
    [11,12],                      
    [11,13],[13,15],              
    [12,14],[14,16],              
    [11,23],[12,24],[23,24]      
];

const HAND_CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],      
    [0,5],[5,6],[6,7],[7,8],      
    [0,9],[9,10],[10,11],[11,12], 
    [0,13],[13,14],[14,15],[15,16], 
    [0,17],[17,18],[18,19],[19,20], 
    [5,9],[9,13],[13,17]          
];

const SpeechToASL = () => {
    const canvasRef = useRef(null);
    const [listening, setListening] = useState(false);
    const [supported, setSupported] = useState(true);
    const [transcript, setTranscript] = useState("");
    const [aslSentenceWords, setAslSentenceWords] = useState([]);
    const [currentSignIndex, setCurrentSignIndex] = useState(-1);
    const [progress, setProgress] = useState(0);

    const recognitionRef = useRef(null);
    const listeningRef = useRef(false);      // avoids stale closure in onend auto-restart
    const finalTextRef = useRef("");          // committed (final) speech so far this session
    const queueRef = useRef([]);              // words waiting to be animated
    const processingRef = useRef(false);
    const committedRef = useRef(0);           // how many transcript words are already queued
    const aslWordsRef = useRef([]);           // mirror of aslSentenceWords for index math
    const avatarScale = 0.72;

    useEffect(() => {
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
    }, []);

    // ── Live speech recognition (Web Speech API — Chrome/Edge) ──────────────
    useEffect(() => {
        const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SR) {
            setSupported(false);
            setTranscript("Live speech needs Chrome or Edge (Web Speech API not found).");
            return;
        }

        const recognition = new SR();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            let interim = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const chunk = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTextRef.current += chunk + " ";
                } else {
                    interim += chunk;
                }
            }
            const full = (finalTextRef.current + interim).trim();
            setTranscript(full);

            // Sign each word as soon as it stabilizes, mid-speech. The last
            // interim word is held back — the recognizer may still revise it.
            // ponytail: earlier interim words can also be revised; we accept the
            // rare mis-sign rather than buffering until the phrase finalizes.
            const words = full ? full.split(/\s+/) : [];
            const stable = interim.trim() ? words.length - 1 : words.length;
            while (committedRef.current < stable) {
                enqueueWord(words[committedRef.current]);
                committedRef.current += 1;
            }
        };

        // Browser stops recognition periodically; restart while the user is still listening.
        recognition.onend = () => {
            if (listeningRef.current) recognition.start();
        };
        recognition.onerror = (e) => {
            console.error("Speech recognition error:", e.error);
            // Fatal errors won't fix themselves on retry — stop the restart loop
            // and tell the user, instead of silently showing "Listening" forever.
            if (e.error === "not-allowed" || e.error === "service-not-allowed") {
                listeningRef.current = false;
                setListening(false);
                setTranscript("Microphone blocked. Click the 🔒/camera icon in the address bar, allow the mic, then Start again.");
            }
        };

        recognitionRef.current = recognition;
        return () => { listeningRef.current = false; recognition.stop(); };
    }, []);

    const toggleListen = () => {
        const rec = recognitionRef.current;
        if (!rec) return;

        if (listening) {
            listeningRef.current = false;
            rec.stop();
            setListening(false);
            return;
        }

        finalTextRef.current = "";
        queueRef.current = [];
        committedRef.current = 0;
        aslWordsRef.current = [];
        setTranscript("Listening — speak now...");
        setAslSentenceWords([]);
        setCurrentSignIndex(-1);
        setProgress(0);
        listeningRef.current = true;
        setListening(true);
        rec.start();
    };

    // Translate one word to ASL and animate it. Words queue so a fast talker
    // doesn't race the avatar — they play back in spoken order.
    const enqueueWord = (word) => {
        if (!word) return;
        queueRef.current.push(word);
        processQueue();
    };

    const processQueue = async () => {
        if (processingRef.current) return;
        processingRef.current = true;
        while (queueRef.current.length > 0) {
            const word = queueRef.current.shift();
            try {
                const res = await fetch("/api/speech_to_asl", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ text: word }),
                });
                const data = await res.json();
                // Filler words (articles, copulas) come back empty — skip them.
                if (data.asl_words && data.asl_words.length > 0) {
                    await animateWords(data.asl_words, data.sequences);
                }
            } catch (error) {
                console.error("Error translating word:", error);
            }
        }
        processingRef.current = false;

        // Queue drained — go idle (don't do this between words, it flickers).
        setCurrentSignIndex(-1);
        setProgress(0);
        const canvas = canvasRef.current;
        if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = "#0f172a";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
    };

    const drawSkeleton = (ctx, frameFeat, w, h) => {
        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, w, h);

        const pose = frameFeat.slice(0, 132);
        const lh = frameFeat.slice(132, 195);
        const rh = frameFeat.slice(195, 258);

        const landmarksToPixels = (x, y, offsetY) => ({
            x: (0.5 + x * avatarScale) * w,
            y: (0.5 + y * avatarScale) * h + offsetY
        });

        const posePts = [];
        for (let i = 0; i < 33; i++) {
            posePts.push(landmarksToPixels(pose[i*4], pose[i*4+1], -50));
        }

        ctx.lineWidth = 4;
        ctx.strokeStyle = "#818cf8";
        POSE_CONNECTIONS.forEach(([a, b]) => {
            if (a < posePts.length && b < posePts.length) {
                ctx.beginPath();
                ctx.moveTo(posePts[a].x, posePts[a].y);
                ctx.lineTo(posePts[b].x, posePts[b].y);
                ctx.stroke();
            }
        });

        ctx.fillStyle = "#c084fc";
        for (let i = 0; i < 25; i++) {
            ctx.beginPath();
            ctx.arc(posePts[i].x, posePts[i].y, 6, 0, 2 * Math.PI);
            ctx.fill();
        }

        const lhPts = [];
        for (let i = 0; i < 21; i++) {
            lhPts.push(landmarksToPixels(lh[i*3], lh[i*3+1], -50));
        }

        ctx.strokeStyle = "#38bdf8";
        HAND_CONNECTIONS.forEach(([a, b]) => {
            if (a < lhPts.length && b < lhPts.length) {
                ctx.beginPath();
                ctx.moveTo(lhPts[a].x, lhPts[a].y);
                ctx.lineTo(lhPts[b].x, lhPts[b].y);
                ctx.stroke();
            }
        });
        
        ctx.fillStyle = "#bae6fd";
        lhPts.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        });

        const rhPts = [];
        for (let i = 0; i < 21; i++) {
            rhPts.push(landmarksToPixels(rh[i*3], rh[i*3+1], -50));
        }

        ctx.strokeStyle = "#f472b6";
        HAND_CONNECTIONS.forEach(([a, b]) => {
            if (a < rhPts.length && b < rhPts.length) {
                ctx.beginPath();
                ctx.moveTo(rhPts[a].x, rhPts[a].y);
                ctx.lineTo(rhPts[b].x, rhPts[b].y);
                ctx.stroke();
            }
        });

        ctx.fillStyle = "#fbcfe8";
        rhPts.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, 5, 0, 2 * Math.PI);
            ctx.fill();
        });
    };

    const animateWords = async (asl_words, sequences) => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;

        // Append — the sentence grows word by word while the user talks.
        const base = aslWordsRef.current.length;
        aslWordsRef.current = [...aslWordsRef.current, ...asl_words];
        setAslSentenceWords(aslWordsRef.current);

        for (let wordIdx = 0; wordIdx < asl_words.length; wordIdx++) {
            const word = asl_words[wordIdx];
            setCurrentSignIndex(base + wordIdx);
            
            const seq = sequences[word];
            if (!seq) {
                ctx.clearRect(0, 0, w, h);
                ctx.fillStyle = "#0f172a";
                ctx.fillRect(0, 0, w, h);
                ctx.fillStyle = "#ef4444";
                ctx.font = "30px Inter";
                ctx.textAlign = "center";
                ctx.fillText(`'${word.toUpperCase()}' not in vocabulary`, w/2, h/2);
                await new Promise(r => setTimeout(r, 1000));
                continue;
            }

            for (let i = 0; i < seq.length; i++) {
                drawSkeleton(ctx, seq[i], w, h);
                setProgress(((i + 1) / seq.length) * 100);
                await new Promise(r => setTimeout(r, 50)); 
            }
            await new Promise(r => setTimeout(r, 300));
        }
    };

    return (
        <div className="speech-container">
            <header className="speech-header">
                <h1>Speech to ASL</h1>
                <p>Speak in English. See the grammar-corrected ASL avatar.</p>
            </header>

            <main className="speech-main">
                <div className="canvas-wrapper">
                    <canvas 
                        ref={canvasRef} 
                        width={900} 
                        height={600} 
                        className="avatar-canvas"
                    />
                    <div className="canvas-hud">
                        <div className="hud-sign">
                            Signing: <span>{currentSignIndex >= 0 ? aslSentenceWords[currentSignIndex].toUpperCase() : "IDLE"}</span>
                        </div>
                        <div className="hud-progress">
                            <div className="hud-progress-bar" style={{width: `${progress}%`}}></div>
                        </div>
                    </div>
                </div>

                <div className="control-board">
                    <button
                        className={`mic-btn ${listening ? 'listening' : ''}`}
                        onClick={toggleListen}
                        disabled={!supported}
                    >
                        {listening ? "Stop Microphone" : "Start Microphone"}
                    </button>
                    
                    <div className="transcription-box">
                        <p className="label">Spoken English</p>
                        <p className="text english-text">{transcript || (listening ? "Listening..." : "Waiting for speech...")}</p>
                    </div>

                    <div className="transcription-box">
                        <p className="label">ASL Grammar Translation</p>
                        <p className="text asl-text">
                            {aslSentenceWords.length > 0 ? (
                                aslSentenceWords.map((word, idx) => (
                                    <span 
                                        key={idx} 
                                        className={idx === currentSignIndex ? "active-word" : ""}
                                        style={{ marginRight: '8px', transition: 'color 0.2s ease' }}
                                    >
                                        {word.toUpperCase()}
                                    </span>
                                ))
                            ) : "Waiting for translation..."}
                        </p>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default SpeechToASL;
