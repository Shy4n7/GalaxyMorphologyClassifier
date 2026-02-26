
import React, { useState, useEffect } from 'react';
import { MOCK_CLASSIFICATIONS } from '../constants';

interface EnsembleFeedProps {
  isRunning?: boolean;
}

const EnsembleFeed: React.FC<EnsembleFeedProps> = ({ isRunning = false }) => {
  const [activeNodes, setActiveNodes] = useState<number[]>([]);

  useEffect(() => {
    if (!isRunning) {
      setActiveNodes([]);
      return;
    }

    const interval = setInterval(() => {
      const newNodes = Array.from({ length: 12 }, () => Math.floor(Math.random() * 48));
      setActiveNodes(newNodes);
    }, 200);

    return () => clearInterval(interval);
  }, [isRunning]);

  return (
    <div className={`bg-surface/60 border rounded-xl p-4 flex flex-col transition-all duration-500 ${isRunning ? 'border-primary/50 shadow-[0_0_30px_rgba(19,91,236,0.1)]' : 'border-primary/10'}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xs font-bold text-slate-300 uppercase tracking-[0.2em] flex items-center gap-2">
          <span className={`material-icons text-sm ${isRunning ? 'text-primary animate-pulse' : 'text-slate-600'}`}>analytics</span>
          Ensemble Classification Live Feed
        </h2>
        <div className="flex gap-4">
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full bg-primary ${isRunning ? 'animate-pulse' : 'opacity-40'}`}></div>
            <span className="text-[9px] text-slate-500">ResNet</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full bg-accent ${isRunning ? 'animate-pulse' : 'opacity-40'}`}></div>
            <span className="text-[9px] text-slate-500">VGG</span>
          </div>
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full bg-emerald-500 ${isRunning ? 'animate-pulse' : 'opacity-40'}`}></div>
            <span className="text-[9px] text-slate-500">Incept</span>
          </div>
        </div>
      </div>

      <div className="flex-1 grid grid-cols-12 gap-6 overflow-hidden">
        <div className="col-span-4 space-y-4">
          {MOCK_CLASSIFICATIONS.map((c, i) => (
            <div key={c.label} className="flex flex-col gap-1 group">
              <div className="flex justify-between items-end">
                <span className={`text-xs font-bold transition-colors ${isRunning && i === 0 ? 'text-white' : 'text-slate-500'}`}>{c.label}</span>
                <span className={`text-xs font-mono transition-colors ${isRunning && i === 0 ? 'text-primary' : 'text-slate-600'}`}>{isRunning ? c.confidence : '0.0'}%</span>
              </div>
              <div className="h-2 bg-white/5 rounded-full flex gap-0.5 overflow-hidden">
                <div 
                  className="h-full bg-primary transition-all duration-1000" 
                  style={{ width: isRunning && i === 0 ? '40%' : '0%' }}
                ></div>
                <div 
                  className="h-full bg-accent transition-all duration-1000 delay-100" 
                  style={{ width: isRunning && i === 0 ? '30%' : '0%' }}
                ></div>
                <div 
                  className="h-full bg-emerald-500 transition-all duration-1000 delay-200" 
                  style={{ width: isRunning && i === 0 ? '28%' : '0%' }}
                ></div>
              </div>
            </div>
          ))}
        </div>

        <div className="col-span-6 bg-background/80 rounded-lg p-3 flex flex-col border border-white/5 relative overflow-hidden">
          <div className="flex items-center justify-between text-[10px] text-slate-500 mb-2 font-bold uppercase tracking-widest relative z-10">
            <span>LAYER ATTENTION MAP</span>
            <span className={`material-icons text-xs ${isRunning ? 'text-primary' : ''}`}>visibility</span>
          </div>
          <div className="flex-1 grid grid-cols-12 grid-rows-4 gap-1 relative z-10">
            {Array.from({ length: 48 }).map((_, i) => (
              <div 
                key={i} 
                className={`rounded-sm transition-all duration-300 ${activeNodes.includes(i) ? 'bg-primary shadow-[0_0_8px_#135bec]' : 'bg-primary/5'}`}
              ></div>
            ))}
          </div>
        </div>

        <div className="col-span-2 flex flex-col justify-center gap-1 border-l border-white/5 pl-4 overflow-hidden">
          <div className="text-[8px] text-slate-500 uppercase font-bold tracking-widest">Ensemble Voting</div>
          <div className={`text-lg font-bold tracking-tighter transition-all ${isRunning ? 'text-white' : 'text-slate-800'}`}>
            {isRunning ? 'GOLDEN' : '---'}
          </div>
          <div className={`text-[10px] font-mono tracking-widest transition-all ${isRunning ? 'text-primary' : 'text-slate-800'}`}>
            {isRunning ? 'MAJORITY WIN' : 'WAITING...'}
          </div>
        </div>
      </div>
    </div>
  );
};

export default EnsembleFeed;
