import React, { useState, useRef, useCallback } from 'react';
import ModelCard from './components/ModelCard';
import EnsembleFeed from './components/EnsembleFeed';
import { INITIAL_MODELS } from './constants';
import { ModelConfig, GalaxyPrediction, GradCAMData } from './types';

const API = 'http://localhost:8000';

const App: React.FC = () => {
  const [models] = useState<ModelConfig[]>(INITIAL_MODELS);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<GalaxyPrediction | null>(null);
  const [gradCAM, setGradCAM] = useState<GradCAMData | null>(null);
  const [isClassifying, setIsClassifying] = useState(false);
  const [showGradCAM, setShowGradCAM] = useState(false);
  const [classifyError, setClassifyError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

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
        top3: data.top3.map((t: { class: string; confidence: number }) => ({ label: t.class, confidence: t.confidence })),
        individualModels: data.individual_models,
        allProbabilities: data.all_probabilities,
      });
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

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background text-slate-200">

      {/* Header */}
      <nav className="relative z-50 flex items-center justify-between px-10 py-5 glass-card border-b border-white/5">
        <div className="flex items-center gap-4">
          <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center text-white shadow-lg shadow-primary/20">
            <span className="material-icons">scatter_plot</span>
          </div>
          <div>
            <span className="text-xl font-black tracking-tight uppercase">
              Galaxy <span className="text-primary">Classifier</span>
            </span>
            <p className="text-[9px] text-slate-500 font-mono tracking-widest uppercase">3-Model Ensemble · 87.97% TTA</p>
          </div>
        </div>

        <div className="flex items-center gap-10 text-[11px] font-semibold tracking-[0.2em] uppercase text-slate-400">
          <span className="text-primary cursor-default">Classify</span>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
            <span className="text-slate-500">Ensemble</span>
            <span className="text-white font-bold">87.97% TTA</span>
          </div>
          <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-[10px] font-mono font-bold">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
            3 MODELS ACTIVE
          </div>
        </div>
      </nav>

      {/* Main */}
      <main className="flex-1 flex p-4 gap-4 overflow-hidden">

        {/* Left: Input Panel */}
        <section className="w-72 flex-shrink-0 flex flex-col gap-4">
          <div className="flex-[2] bg-surface/40 border border-primary/20 rounded-xl p-4 flex flex-col relative overflow-hidden">
            <h2 className="text-xs font-bold text-primary uppercase tracking-widest mb-4">Input Image</h2>

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

            <div className="mt-4 grid grid-cols-2 gap-2">
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
        <section className="flex-1 flex flex-col gap-4 overflow-hidden min-w-0">
          <div className="flex-1 grid grid-cols-3 gap-4 overflow-hidden">
            {models.map(m => (
              <ModelCard key={m.id} model={m} />
            ))}
          </div>
          <EnsembleFeed isRunning={!!prediction} prediction={prediction} gradCAM={gradCAM} showGradCAM={showGradCAM} />
        </section>
      </main>
    </div>
  );
};

export default App;
