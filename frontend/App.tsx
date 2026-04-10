
import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import ModelCard from './components/ModelCard';
import EnsembleFeed from './components/EnsembleFeed';
import { INITIAL_MODELS } from './constants';
import { ViewMode, ModelConfig, InferenceStats } from './types';
import { getEnsembleSuggestion } from './services/geminiService';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const App: React.FC = () => {
  const [viewMode, setViewMode] = useState<ViewMode>('SANDBOX');
  const [models, setModels] = useState<ModelConfig[]>(INITIAL_MODELS);
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState<InferenceStats>({
    throughput: '0.0',
    gpuLoad: 0,
    latency: 0,
    sessionTime: '00:00:00:00',
    inferences: 0,
  });
  const [aiSuggestion, setAiSuggestion] = useState<string>('');
  const [isAskingAi, setIsAskingAi] = useState(false);
  const [useCamera, setUseCamera] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const startTimeRef = useRef<number>(Date.now());

  // Camera handling
  useEffect(() => {
    let stream: MediaStream | null = null;
    const startCamera = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Camera access denied", err);
        setUseCamera(false);
      }
    };

    if (useCamera) {
      startCamera();
    } else if (stream) {
      (stream as MediaStream).getTracks().forEach(track => track.stop());
    }

    return () => {
      if (stream) {
        (stream as MediaStream).getTracks().forEach(track => track.stop());
      }
    };
  }, [useCamera]);

  // Simulation loop
  useEffect(() => {
    let interval: number;
    if (isRunning) {
      interval = window.setInterval(() => {
        setStats(prev => {
          const elapsed = Date.now() - startTimeRef.current;
          const h = Math.floor(elapsed / 3600000).toString().padStart(2, '0');
          const m = Math.floor((elapsed % 3600000) / 60000).toString().padStart(2, '0');
          const s = Math.floor((elapsed % 60000) / 1000).toString().padStart(2, '0');
          const ms = Math.floor((elapsed % 1000) / 10).toString().padStart(2, '0');

          return {
            ...prev,
            throughput: (140 + Math.random() * 10).toFixed(1),
            gpuLoad: Math.floor(40 + Math.random() * 15),
            latency: Math.floor(12 + Math.random() * 5),
            inferences: prev.inferences + Math.floor(Math.random() * 5),
            sessionTime: `${h}:${m}:${s}:${ms}`
          };
        });
      }, 100);
    } else {
      setStats(prev => ({ ...prev, throughput: '0.0', gpuLoad: 5 }));
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleAskAi = async () => {
    setIsAskingAi(true);
    const suggestion = await getEnsembleSuggestion(models, 98.4);
    setAiSuggestion(suggestion || 'Suggestion unavailable');
    setIsAskingAi(false);
  };

  const updateWeight = (id: string, weight: number) => {
    setModels(prev => prev.map(m => m.id === id ? { ...m, weight } : m));
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-slate-200">
      <Header
        viewMode={viewMode}
        setViewMode={setViewMode}
        fps={stats.throughput}
        gpuLoad={stats.gpuLoad}
      />

      <main className="flex-1 flex flex-col p-4 gap-4 overflow-hidden relative">
        {viewMode === 'SANDBOX' && (
          <div className="flex-1 grid grid-cols-12 gap-4 overflow-hidden">
            {/* Left: Input Panel */}
            <section className="col-span-3 flex flex-col gap-4">
              <div className="flex-[2] bg-surface/40 border border-primary/20 rounded-xl p-4 flex flex-col relative overflow-hidden group">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xs font-bold text-primary uppercase tracking-widest">Input Source</h2>
                  <div className="flex bg-background/60 rounded-lg p-1 border border-white/5">
                    <button
                      onClick={() => setUseCamera(false)}
                      className={`p-1 rounded ${!useCamera ? 'bg-primary text-white' : 'text-slate-500'}`}
                    >
                      <span className="material-icons text-xs">image</span>
                    </button>
                    <button
                      onClick={() => setUseCamera(true)}
                      className={`p-1 rounded ${useCamera ? 'bg-primary text-white' : 'text-slate-500'}`}
                    >
                      <span className="material-icons text-xs">videocam</span>
                    </button>
                  </div>
                </div>
                <div className="flex-1 border-2 border-dashed border-primary/30 rounded-lg relative flex items-center justify-center bg-background/40 overflow-hidden">
                  {useCamera ? (
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="absolute inset-0 w-full h-full object-cover"
                    />
                  ) : (
                    <img
                      alt="Source"
                      className="absolute inset-0 w-full h-full object-cover opacity-60 grayscale group-hover:grayscale-0 transition-all"
                      src="https://images.unsplash.com/photo-1543466835-00a7907e9de1?auto=format&fit=crop&q=80&w=800"
                    />
                  )}
                  {isRunning && <div className="absolute inset-0 scan-line z-10"></div>}
                  {!isRunning && !useCamera && (
                    <div className="z-20 text-center">
                      <button className="bg-primary hover:bg-primary/80 text-white rounded-full p-3 shadow-lg mb-2">
                        <span className="material-icons">cloud_upload</span>
                      </button>
                      <p className="text-xs text-slate-400">Update dataset</p>
                    </div>
                  )}
                </div>
                <div className="mt-4 space-y-3">
                  <div className="flex justify-between items-center text-[10px] uppercase font-bold text-slate-500">
                    <span>Preprocessing</span>
                    <span className={isRunning ? "text-primary animate-pulse" : "text-slate-700"}>
                      {isRunning ? "ACTIVE" : "STANDBY"}
                    </span>
                  </div>
                  <div className="h-1 bg-primary/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-500"
                      style={{ width: isRunning ? '85%' : '0%', boxShadow: '0 0 8px rgba(19,91,236,0.6)' }}
                    ></div>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-background/80 p-2 rounded border border-primary/10">
                      <p className="text-[8px] text-slate-500 uppercase">Rescale</p>
                      <p className="text-xs font-mono">224x224</p>
                    </div>
                    <div className="bg-background/80 p-2 rounded border border-primary/10">
                      <p className="text-[8px] text-slate-500 uppercase">Format</p>
                      <p className="text-xs font-mono">RGB_32</p>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex-1 bg-surface/40 border border-primary/20 rounded-xl p-4 flex flex-col justify-between">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">System Controls</h2>
                <div className="space-y-2">
                  <button
                    onClick={() => setIsRunning(!isRunning)}
                    className={`w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all ${isRunning ? 'bg-red-500/20 text-red-500 border border-red-500/40' : 'bg-primary text-white shadow-lg shadow-primary/20'}`}
                  >
                    <span className="material-icons">{isRunning ? 'stop' : 'play_arrow'}</span>
                    {isRunning ? 'STOP INFERENCE' : 'RUN ENSEMBLE'}
                  </button>
                  <button
                    onClick={handleAskAi}
                    disabled={isAskingAi}
                    className="w-full bg-surface/80 border border-primary/40 text-primary py-3 rounded-lg font-bold flex items-center justify-center gap-2 hover:bg-primary/10 transition-all disabled:opacity-50"
                  >
                    <span className="material-icons">{isAskingAi ? 'sync' : 'psychology'}</span>
                    {isAskingAi ? 'ANALYZING...' : 'AI SUGGESTION'}
                  </button>
                </div>
              </div>
            </section>

            {/* Right: Model Cards + Feed */}
            <section className="col-span-9 flex flex-col gap-4 overflow-hidden">
              <div className="flex-1 grid grid-cols-4 gap-4 overflow-hidden">
                {models.map(m => (
                  <ModelCard key={m.id} model={m} onClick={() => {}} />
                ))}
              </div>
              <EnsembleFeed isRunning={isRunning} />
            </section>
          </div>
        )}

        {viewMode === 'LAB_GRID' && (
          <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-1 bg-primary/10 p-1 rounded-xl overflow-hidden relative">
            {models.map((m, i) => (
              <div key={m.id} className={`relative bg-background p-8 flex flex-col items-center justify-center transition-all duration-700 ${i % 2 !== 0 ? 'border-l border-primary/10' : ''} ${i > 1 ? 'border-t border-primary/10' : ''}`}>
                <div className="absolute top-4 left-6 flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full bg-primary ${isRunning ? 'animate-pulse' : ''}`}></span>
                  <span className="text-xs font-bold text-primary uppercase tracking-widest">{m.name}</span>
                </div>
                <div className="w-full max-w-sm h-48 border border-primary/20 rounded-xl bg-surface/20 backdrop-blur-sm flex items-center justify-center gap-4 relative overflow-hidden group">
                  <div className={`absolute inset-0 bg-gradient-to-r from-transparent via-primary/5 to-transparent -translate-x-full ${isRunning ? 'animate-[scan_2s_infinite_linear]' : ''}`}></div>
                  <div className="w-12 h-12 rounded border-2 border-primary bg-primary/20 flex items-center justify-center z-10">
                    <span className="material-icons text-primary">hub</span>
                  </div>
                  <div className="w-8 h-px bg-primary/30 z-10"></div>
                  <div className={`w-12 h-12 rounded border-2 border-primary/50 bg-primary/10 flex items-center justify-center z-10 ${isRunning ? 'node-pulse' : ''}`}>
                    <span className="material-icons text-primary/70">auto_graph</span>
                  </div>
                  <div className="w-8 h-px bg-primary/30 z-10"></div>
                  <div className="w-12 h-12 rounded border-2 border-primary bg-primary/20 flex items-center justify-center z-10">
                    <span className="material-icons text-primary">science</span>
                  </div>
                </div>
                <div className="mt-6 font-mono text-[10px] text-slate-500 flex gap-4">
                  <span>LOSS: {(0.02 + Math.random() * 0.01).toFixed(4)}</span>
                  <span>ACC: {m.accuracy}%</span>
                </div>
              </div>
            ))}
            {/* Central Hub Overlay */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-40 bg-background border-2 border-primary rounded-full w-48 h-48 flex flex-col items-center justify-center shadow-[0_0_40px_rgba(19,91,236,0.3)] group cursor-pointer hover:scale-105 transition-all">
              <div className="absolute inset-0 rounded-full border border-primary/20 animate-ping"></div>
              <span className="text-[10px] font-bold text-primary uppercase mb-1">Ensemble Confidence</span>
              <span className="text-4xl font-bold">98.4<span className="text-lg opacity-50">%</span></span>
              <div className="mt-2 px-3 py-1 bg-primary text-white text-[10px] font-bold rounded-full uppercase tracking-tighter shadow-lg shadow-primary/40">GOLDEN RETRIEVER</div>
            </div>
          </div>
        )}

        {viewMode === 'ANALYTICS' && (
          <div className="flex-1 grid grid-cols-12 gap-4 overflow-hidden">
            <div className="col-span-8 flex flex-col gap-4">
              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex flex-col h-1/3">
                <h2 className="text-sm font-bold text-white mb-4 uppercase tracking-widest flex items-center gap-2">
                  <span className="material-icons text-primary">list_alt</span>
                  Ensemble Weights Matrix
                </h2>
                <div className="grid grid-cols-4 gap-4">
                  {models.map(m => (
                    <div key={m.id} className="bg-background/80 p-4 rounded-lg border border-white/5 group hover:border-primary/40 transition-all">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-slate-500 mb-2">
                        <span className="group-hover:text-white transition-colors">{m.name}</span>
                        <span className="text-primary font-mono">{m.weight.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0" max="1" step="0.01"
                        value={m.weight}
                        onChange={(e) => updateWeight(m.id, parseFloat(e.target.value))}
                        className="w-full h-1 bg-white/10 rounded-full appearance-none accent-primary cursor-pointer hover:accent-accent transition-all"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex-1 flex flex-col">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-sm font-bold text-white flex items-center gap-2 uppercase tracking-widest">
                    <span className="material-icons text-primary">show_chart</span>
                    Weight Sensitivity Analysis
                  </h2>
                  <div className="flex gap-2">
                    <span className="text-[10px] bg-primary/10 text-primary px-2 py-0.5 rounded border border-primary/20">GLOBAL OPTIMUM</span>
                  </div>
                </div>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[
                      { ratio: '10/90', accuracy: 91.2 },
                      { ratio: '30/70', accuracy: 92.8 },
                      { ratio: '50/50', accuracy: 93.4 },
                      { ratio: '70/30', accuracy: 95.1 },
                      { ratio: '90/10', accuracy: 94.2 },
                      { ratio: '100/0', accuracy: 93.9 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                      <XAxis dataKey="ratio" stroke="#475569" fontSize={10} axisLine={false} tickLine={false} />
                      <YAxis stroke="#475569" fontSize={10} axisLine={false} tickLine={false} domain={[90, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#161b2a', border: '1px solid #135bec30', borderRadius: '8px' }}
                        itemStyle={{ color: '#135bec', fontSize: '12px', fontWeight: 'bold' }}
                      />
                      <Line
                        type="monotone"
                        dataKey="accuracy"
                        stroke="#135bec"
                        strokeWidth={3}
                        dot={{ fill: '#135bec', r: 4 }}
                        activeDot={{ r: 6, stroke: '#fff' }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="col-span-4 flex flex-col gap-4">
              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex-1 flex flex-col overflow-hidden">
                <h2 className="text-sm font-bold text-white mb-4 uppercase tracking-widest flex items-center gap-2">
                  <span className="material-icons text-primary">psychology</span>
                  AI Architect Suggestions
                </h2>
                <div className="flex-1 overflow-y-auto custom-scrollbar pr-2">
                  {aiSuggestion ? (
                    <div className="text-sm text-slate-300 leading-relaxed whitespace-pre-wrap font-mono bg-background/40 p-4 rounded-lg border border-white/5">
                      {aiSuggestion}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center h-full text-slate-500 gap-4 text-center">
                      <div className="w-16 h-16 rounded-full border-2 border-dashed border-slate-700 flex items-center justify-center">
                        <span className="material-icons text-3xl opacity-20">insights</span>
                      </div>
                      <p className="text-xs uppercase tracking-widest max-w-[200px]">Run system controls to generate generative optimization insights</p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Loading Overlay for AI */}
        {isAskingAi && (
          <div className="absolute inset-0 z-[100] bg-background/80 backdrop-blur-md flex items-center justify-center">
            <div className="flex flex-col items-center gap-6 max-w-sm text-center">
              <div className="relative">
                <div className="w-24 h-24 border-4 border-primary/20 rounded-full"></div>
                <div className="w-24 h-24 border-t-4 border-primary rounded-full absolute top-0 animate-spin"></div>
                <span className="material-icons text-primary text-4xl absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2">bolt</span>
              </div>
              <div className="space-y-2">
                <h3 className="text-xl font-bold uppercase tracking-[0.2em] text-white">Synthesizing Optimization</h3>
                <p className="text-xs text-slate-500 font-medium">Gemini is analyzing the ensemble architecture and weight distribution for maximum precision...</p>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="bg-surface border-t border-primary/20 px-6 py-2 flex justify-between items-center z-50">
        <div className="flex gap-6 items-center">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Session Time</span>
            <span className="text-xs font-mono text-slate-300">{stats.sessionTime}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Inferences</span>
            <span className="text-xs font-mono text-slate-300">{stats.inferences.toLocaleString()}</span>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="material-icons text-[10px] text-primary">cloud_done</span>
            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-widest">Cloud Sync Active</span>
          </div>
          <div className="w-px h-4 bg-white/10 mx-2"></div>
          <div className="flex items-center gap-3">
            <span className="material-icons text-slate-500 hover:text-primary cursor-pointer text-sm">help_outline</span>
            <span className="material-icons text-slate-500 hover:text-primary cursor-pointer text-sm">notifications</span>
            <div className="w-6 h-6 rounded-full bg-slate-800 border border-primary/40 flex items-center justify-center cursor-pointer hover:border-primary">
              <span className="text-[8px] font-bold">JD</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
