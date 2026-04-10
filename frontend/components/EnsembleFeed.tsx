import React, { useState, useEffect } from 'react';
import { GalaxyPrediction, GradCAMData } from '../types';

interface EnsembleFeedProps {
  isRunning?: boolean;
  prediction?: GalaxyPrediction | null;
  gradCAM?: GradCAMData | null;
  showGradCAM?: boolean;
}

const MODEL_INDICATORS = [
  { key: 'convnext',   label: 'ConvNeXt',  color: 'bg-red-500' },
  { key: 'resnext50',  label: 'ResNeXt',   color: 'bg-emerald-500' },
  { key: 'densenet161',label: 'DenseNet',  color: 'bg-orange-500' },
];

const EnsembleFeed: React.FC<EnsembleFeedProps> = ({
  isRunning = false,
  prediction = null,
  gradCAM = null,
  showGradCAM = false,
}) => {
  const [activeNodes, setActiveNodes] = useState<number[]>([]);

  useEffect(() => {
    if (!isRunning) { setActiveNodes([]); return; }
    const interval = setInterval(() => {
      setActiveNodes(Array.from({ length: 48 }, () => Math.floor(Math.random() * 48)));
    }, 200);
    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className={`bg-surface/60 border rounded-xl p-4 flex flex-col transition-all duration-500 ${isRunning ? 'border-primary/50 shadow-[0_0_30px_rgba(19,91,236,0.1)]' : 'border-primary/10'}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2">
          <span className={`material-icons text-sm ${isRunning ? 'text-primary animate-pulse' : 'text-slate-600'}`}>analytics</span>
          Ensemble Classification Feed
        </h2>
        <div className="flex gap-4">
          {MODEL_INDICATORS.map(m => (
            <div key={m.key} className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${m.color} ${prediction ? 'animate-pulse' : 'opacity-40'}`}></div>
              <span className="text-[9px] text-slate-500">{m.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
        {/* Top 3 Predictions */}
        <div className="col-span-4 space-y-4">
          {prediction ? (
            prediction.top3.map((c, i) => (
              <div key={c.label} className="flex flex-col gap-1">
                <div className="flex justify-between items-end">
                  <span className={`text-xs font-bold ${i === 0 ? 'text-white' : 'text-slate-500'}`}>{c.label}</span>
                  <span className={`text-xs font-mono ${i === 0 ? 'text-primary' : 'text-slate-600'}`}>{c.confidence.toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-1000 ${i === 0 ? 'bg-primary' : i === 1 ? 'bg-accent' : 'bg-emerald-500'}`}
                    style={{ width: `${c.confidence}%` }}
                  ></div>
                </div>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-slate-600">
              <span className="material-icons text-4xl opacity-20">pending</span>
              <p className="text-xs mt-2 uppercase tracking-wider">Awaiting Classification</p>
            </div>
          )}
        </div>

        {/* Grad-CAM / Attention Map */}
        <div className="col-span-6 bg-background/80 rounded-lg p-3 flex flex-col border border-white/5 relative overflow-hidden">
          <div className="flex items-center justify-between text-[10px] text-slate-500 mb-2 font-bold uppercase tracking-widest">
            <span>{showGradCAM && gradCAM ? 'GRAD-CAM VISUALIZATION' : 'LAYER ATTENTION MAP'}</span>
            <span className={`material-icons text-xs ${isRunning || gradCAM ? 'text-primary' : ''}`}>visibility</span>
          </div>
          {showGradCAM && gradCAM ? (
            <div className="flex-1 flex items-center justify-center">
              <img src={`data:image/png;base64,${gradCAM.gridImage}`} alt="Grad-CAM" className="max-w-full max-h-full object-contain rounded" />
            </div>
          ) : (
            <div className="flex-1 grid grid-cols-12 grid-rows-4 gap-1">
              {Array.from({ length: 48 }).map((_, i) => (
                <div key={i} className={`rounded-sm transition-all duration-300 ${activeNodes.includes(i) ? 'bg-primary shadow-[0_0_8px_#135bec]' : 'bg-primary/5'}`}></div>
              ))}
            </div>
          )}
        </div>

        {/* Ensemble Result */}
        <div className="col-span-2 flex flex-col justify-center gap-1 border-l border-white/5 pl-4 overflow-hidden">
          <div className="text-[8px] text-slate-500 uppercase font-bold tracking-widest">Ensemble Result</div>
          <div className={`text-lg font-bold tracking-tighter transition-all ${prediction ? 'text-white' : 'text-slate-800'}`}>
            {prediction ? prediction.prediction.split(' ')[0] : '---'}
          </div>
          <div className={`text-[10px] font-mono tracking-widest transition-all ${prediction ? 'text-primary' : 'text-slate-800'}`}>
            {prediction ? `${prediction.confidence.toFixed(1)}% CONF` : 'WAITING...'}
          </div>
          {prediction && (
            <div className="mt-3 space-y-1">
              {MODEL_INDICATORS.map((m, i) => {
                const res = prediction.individualModels?.[m.key];
                return res ? (
                  <div key={m.key} className="text-[8px] font-mono text-slate-500">
                    <span className={`inline-block w-1.5 h-1.5 rounded-full mr-1 ${m.color}`}></span>
                    {res.confidence.toFixed(1)}%
                  </div>
                ) : null;
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EnsembleFeed;
