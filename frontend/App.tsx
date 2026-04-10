import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import ModelCard from './components/ModelCard';
import EnsembleFeed from './components/EnsembleFeed';
import { INITIAL_MODELS, ENSEMBLE_ACCURACY } from './constants';
import { ViewMode, ModelConfig, InferenceStats, GalaxyPrediction, GradCAMData } from './types';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const API = 'http://localhost:8080';

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

  // Classification state
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<GalaxyPrediction | null>(null);
  const [gradCAM, setGradCAM] = useState<GradCAMData | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const startTimeRef = useRef<number>(Date.now());
  const fileInputRef = useRef<HTMLInputElement>(null);

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
            sessionTime: `${h}:${m}:${s}:${ms}`,
          };
        });
      }, 100);
    } else {
      setStats(prev => ({ ...prev, throughput: '0.0', gpuLoad: 5 }));
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  const handleFileSelect = useCallback((file: File) => {
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setPrediction(null);
    setGradCAM(null);
    setClassifyError(null);
    setShowGradCAM(false);
  }, []);

  const handleClassify = useCallback(async () => {
    if (!selectedFile) return;
    setIsClassifying(true);
    setClassifyError(null);
    try {
      const fd = new FormData();
      fd.append('image', selectedFile);
      const res = await fetch(`${API}/api/predict`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Prediction failed');
      const data = await res.json();
      setPrediction({
        prediction: data.prediction,
        confidence: data.confidence,
        top3: data.top3.map((t: any) => ({ label: t.class, confidence: t.confidence })),
        individualModels: data.individual_models,
        allProbabilities: data.all_probabilities,
      });
      setIsRunning(true);
    } catch (e: unknown) {
      setClassifyError(e instanceof Error ? e.message : 'Classification failed');
    } finally {
      setIsClassifying(false);
    }
  }, [selectedFile]);

  const handleGradCAM = useCallback(async () => {
    if (!selectedFile) return;
    setIsClassifying(true);
    try {
      const fd = new FormData();
      fd.append('image', selectedFile);
      const res = await fetch(`${API}/api/gradcam`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error('Grad-CAM failed');
      const data = await res.json();
      setGradCAM({ gridImage: data.grid, predictedClass: data.predicted_class, confidence: data.confidence });
      setShowGradCAM(true);
    } catch (e: unknown) {
      setClassifyError(e instanceof Error ? e.message : 'Grad-CAM failed');
    } finally {
      setIsClassifying(false);
    }
  }, [selectedFile]);

  const updateWeight = (id: string, weight: number) => {
    setModels(prev => prev.map(m => m.id === id ? { ...m, weight } : m));
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-slate-200">
      <Header />

      <main className="flex-1 flex flex-col p-4 gap-4 overflow-hidden relative">

        {/* SANDBOX VIEW */}
        {viewMode === 'SANDBOX' && (
          <div className="flex-1 grid grid-cols-12 gap-4 overflow-hidden">

            {/* Left: Input Panel */}
            <section className="col-span-3 flex flex-col gap-4">
              <div className="flex-[2] bg-surface/40 border border-primary/20 rounded-xl p-4 flex flex-col relative overflow-hidden">
                <h2 className="text-xs font-bold text-primary uppercase tracking-widest mb-4">Input Image</h2>

                {/* Drop Zone */}
                <div
                  className="flex-1 border-2 border-dashed border-primary/30 rounded-lg relative flex items-center justify-center bg-background/40 overflow-hidden cursor-pointer"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={e => e.preventDefault()}
                  onDrop={e => { e.preventDefault(); if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]); }}
                >
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    className="hidden"
                    onChange={e => { if (e.target.files?.[0]) handleFileSelect(e.target.files[0]); }}
                  />
                  {previewUrl ? (
                    <img src={previewUrl} alt="Preview" className="absolute inset-0 w-full h-full object-cover" />
                  ) : (
                    <div className="z-20 text-center p-4">
                      <div className="bg-primary/10 border border-primary/30 rounded-full p-3 mb-2 mx-auto w-fit">
                        <span className="material-icons text-primary">cloud_upload</span>
                      </div>
                      <p className="text-xs text-slate-400">Drop galaxy image here</p>
                      <p className="text-[10px] text-slate-600 mt-1">or click to browse</p>
                    </div>
                  )}
                  {isClassifying && (
                    <div className="absolute inset-0 bg-background/60 flex items-center justify-center z-10">
                      <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                    </div>
                  )}
                </div>

                {/* Preprocessing info */}
                <div className="mt-4 space-y-2">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="bg-background/80 p-2 rounded border border-primary/10">
                      <p className="text-[8px] text-slate-500 uppercase">Rescale</p>
                      <p className="text-xs font-mono">224x224</p>
                    </div>
                    <div className="bg-background/80 p-2 rounded border border-primary/10">
                      <p className="text-[8px] text-slate-500 uppercase">TTA Views</p>
                      <p className="text-xs font-mono">8</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Controls */}
              <div className="flex-1 bg-surface/40 border border-primary/20 rounded-xl p-4 flex flex-col justify-between">
                <h2 className="text-xs font-bold text-slate-400 uppercase tracking-widest">Controls</h2>
                {classifyError && (
                  <p className="text-[10px] text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">{classifyError}</p>
                )}
                <div className="space-y-2">
                  <button
                    onClick={handleClassify}
                    disabled={!selectedFile || isClassifying}
                    className="w-full py-3 rounded-lg font-bold flex items-center justify-center gap-2 transition-all bg-primary text-white shadow-lg shadow-primary/20 disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <span className="material-icons">{isClassifying ? 'sync' : 'play_arrow'}</span>
                    {isClassifying ? 'CLASSIFYING...' : 'RUN ENSEMBLE'}
                  </button>
                  <button
                    onClick={handleGradCAM}
                    disabled={!selectedFile || isClassifying}
                    className="w-full bg-surface/80 border border-primary/40 text-primary py-3 rounded-lg font-bold flex items-center justify-center gap-2 hover:bg-primary/10 transition-all disabled:opacity-40 disabled:cursor-not-allowed"
                  >
                    <span className="material-icons">visibility</span>
                    GRAD-CAM
                  </button>
                  {showGradCAM && (
                    <button
                      onClick={() => { setShowGradCAM(false); setGradCAM(null); }}
                      className="w-full bg-surface/40 border border-white/10 text-slate-400 py-2 rounded-lg text-xs font-bold hover:bg-surface/80 transition-all"
                    >
                      HIDE GRAD-CAM
                    </button>
                  )}
                </div>
              </div>
            </section>

            {/* Right: 3 Model Cards + Feed */}
            <section className="col-span-9 flex flex-col gap-4 overflow-hidden">
              <div className="flex-1 grid grid-cols-3 gap-4 overflow-hidden">
                {models.map(m => (
                  <ModelCard key={m.id} model={m} />
                ))}
              </div>
              <EnsembleFeed isRunning={isRunning} prediction={prediction} gradCAM={gradCAM} showGradCAM={showGradCAM} />
            </section>
          </div>
        )}

        {/* LAB_GRID VIEW */}
        {viewMode === 'LAB_GRID' && (
          <div className="flex-1 grid grid-cols-3 gap-1 bg-primary/10 p-1 rounded-xl overflow-hidden relative">
            {models.map((m, i) => (
              <div key={m.id} className={`relative bg-background p-8 flex flex-col items-center justify-center transition-all duration-700 ${i > 0 ? 'border-l border-primary/10' : ''}`}>
                <div className="absolute top-4 left-6 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                  <span className="text-xs font-bold text-primary uppercase tracking-widest">{m.name}</span>
                </div>
                <div className="w-full max-w-sm h-48 border border-primary/20 rounded-xl bg-surface/20 backdrop-blur-sm flex items-center justify-center gap-4 relative overflow-hidden">
                  <div className="w-12 h-12 rounded border-2 border-primary bg-primary/20 flex items-center justify-center z-10">
                    <span className="material-icons text-primary">hub</span>
                  </div>
                  <div className="w-8 h-px bg-primary/30 z-10"></div>
                  <div className="w-12 h-12 rounded border-2 border-primary/50 bg-primary/10 flex items-center justify-center z-10 node-pulse">
                    <span className="material-icons text-primary/70">auto_graph</span>
                  </div>
                  <div className="w-8 h-px bg-primary/30 z-10"></div>
                  <div className="w-12 h-12 rounded border-2 border-primary bg-primary/20 flex items-center justify-center z-10">
                    <span className="material-icons text-primary">science</span>
                  </div>
                </div>
                <div className="mt-6 font-mono text-[10px] text-slate-500 flex gap-4">
                  <span>ACC: {m.accuracy}%</span>
                  <span>W: {m.weight.toFixed(3)}</span>
                </div>
              </div>
            ))}

            {/* Central Hub Overlay */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-40 bg-background border-2 border-primary rounded-full w-48 h-48 flex flex-col items-center justify-center shadow-[0_0_40px_rgba(19,91,236,0.3)] cursor-pointer hover:scale-105 transition-all">
              <div className="absolute inset-0 rounded-full border border-primary/20 animate-ping"></div>
              <span className="text-[10px] font-bold text-primary uppercase mb-1">Ensemble TTA</span>
              <span className="text-4xl font-bold">87<span className="text-lg opacity-50">.97%</span></span>
              <div className="mt-2 px-3 py-1 bg-primary text-white text-[10px] font-bold rounded-full uppercase tracking-tighter shadow-lg shadow-primary/40">
                {prediction ? prediction.prediction.split(' ')[0] : 'READY'}
              </div>
            </div>
          </div>
        )}

        {/* ANALYTICS VIEW */}
        {viewMode === 'ANALYTICS' && (
          <div className="flex-1 grid grid-cols-12 gap-4 overflow-hidden">
            <div className="col-span-8 flex flex-col gap-4">
              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex flex-col h-1/3">
                <h2 className="text-sm font-bold text-white mb-4 uppercase tracking-widest flex items-center gap-2">
                  <span className="material-icons text-primary">list_alt</span>
                  Ensemble Weight Matrix
                </h2>
                <div className="grid grid-cols-3 gap-4">
                  {models.map(m => (
                    <div key={m.id} className="bg-background/80 p-4 rounded-lg border border-white/5 group hover:border-primary/40 transition-all">
                      <div className="flex justify-between text-[10px] uppercase font-bold text-slate-500 mb-2">
                        <span className="group-hover:text-white transition-colors truncate">{m.name}</span>
                        <span className="text-primary font-mono ml-2">{m.weight.toFixed(3)}</span>
                      </div>
                      <input
                        type="range" min="0" max="1" step="0.01"
                        value={m.weight}
                        onChange={e => updateWeight(m.id, parseFloat(e.target.value))}
                        className="w-full h-1 bg-white/10 rounded-full appearance-none accent-primary cursor-pointer"
                      />
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex-1 flex flex-col">
                <h2 className="text-sm font-bold text-white flex items-center gap-2 uppercase tracking-widest mb-6">
                  <span className="material-icons text-primary">show_chart</span>
                  Per-Class Accuracy (TTA)
                </h2>
                <div className="flex-1 min-h-0">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={[
                      { cls: 'Disturbed', accuracy: 65.4 },
                      { cls: 'Merging',   accuracy: 93.2 },
                      { cls: 'Round',     accuracy: 95.7 },
                      { cls: 'In-between',accuracy: 91.8 },
                      { cls: 'Cigar',     accuracy: 94.0 },
                      { cls: 'Barred',    accuracy: 84.4 },
                      { cls: 'Tight Sp.', accuracy: 87.2 },
                      { cls: 'Loose Sp.', accuracy: 75.4 },
                      { cls: 'Edge-on',   accuracy: 96.7 },
                      { cls: 'Edge+Bulge',accuracy: 95.4 },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                      <XAxis dataKey="cls" stroke="#475569" fontSize={9} axisLine={false} tickLine={false} />
                      <YAxis stroke="#475569" fontSize={10} axisLine={false} tickLine={false} domain={[60, 100]} />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#161b2a', border: '1px solid #135bec30', borderRadius: '8px' }}
                        itemStyle={{ color: '#135bec', fontSize: '12px', fontWeight: 'bold' }}
                        formatter={(v: number) => [`${v}%`, 'Accuracy']}
                      />
                      <Line type="monotone" dataKey="accuracy" stroke="#135bec" strokeWidth={3}
                        dot={{ fill: '#135bec', r: 4 }} activeDot={{ r: 6, stroke: '#fff' }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            <div className="col-span-4 flex flex-col gap-4">
              <div className="bg-surface/40 border border-primary/20 rounded-xl p-6 flex-1 flex flex-col">
                <h2 className="text-sm font-bold text-white mb-6 uppercase tracking-widest flex items-center gap-2">
                  <span className="material-icons text-primary">insights</span>
                  Model Summary
                </h2>
                <div className="space-y-4">
                  {models.map(m => (
                    <div key={m.id} className="bg-background/60 p-4 rounded-lg border border-white/5">
                      <div className="flex justify-between items-center mb-2">
                        <span className="text-xs font-bold text-white">{m.name}</span>
                        <span className="text-xs font-mono text-primary">{m.accuracy}%</span>
                      </div>
                      <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
                        <div className="h-full bg-primary rounded-full" style={{ width: `${m.accuracy}%`, backgroundColor: m.color }}></div>
                      </div>
                      <div className="flex justify-between mt-2 text-[9px] text-slate-500 font-mono">
                        <span>{m.params} params</span>
                        <span>{m.architecture}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* View Mode Tabs */}
      <div className="flex justify-center gap-1 pb-2 z-50">
        {(['SANDBOX', 'LAB_GRID', 'ANALYTICS'] as ViewMode[]).map(v => (
          <button
            key={v}
            onClick={() => setViewMode(v)}
            className={`px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest rounded-full transition-all ${viewMode === v ? 'bg-primary text-white' : 'text-slate-500 hover:text-white'}`}
          >
            {v.replace('_', ' ')}
          </button>
        ))}
      </div>

      <footer className="bg-surface border-t border-primary/20 px-6 py-2 flex justify-between items-center z-50">
        <div className="flex gap-6 items-center">
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Session</span>
            <span className="text-xs font-mono text-slate-300">{stats.sessionTime}</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Inferences</span>
            <span className="text-xs font-mono text-slate-300">{stats.inferences.toLocaleString()}</span>
          </div>
          {prediction && (
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-wider">Last</span>
              <span className="text-xs font-mono text-primary">{prediction.prediction} ({prediction.confidence.toFixed(1)}%)</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-4">
          <a href="http://localhost:8080" target="_blank" className="text-[10px] text-primary hover:text-white font-bold uppercase tracking-widest transition-colors">
            Open Classifier UI
          </a>
          <div className="w-px h-4 bg-white/10"></div>
          <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></span>
            <span className="text-[10px] text-slate-400 uppercase font-bold tracking-widest">Ensemble Online</span>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default App;
