
import React from 'react';
import { ViewMode } from '../types';

interface HeaderProps {
  viewMode: ViewMode;
  setViewMode: (v: ViewMode) => void;
  fps: string;
  gpuLoad: number;
}

const Header: React.FC<HeaderProps> = ({ viewMode, setViewMode, fps, gpuLoad }) => {
  return (
    <header className="border-b border-primary/20 bg-background/50 backdrop-blur-md px-6 py-4 flex justify-between items-center z-50">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center text-white shadow-[0_0_15px_rgba(19,91,236,0.4)]">
          <span className="material-icons">Hub</span>
        </div>
        <div>
          <h1 className="text-xl font-bold tracking-tight uppercase">
            Ensemble <span className="text-primary">Sandbox</span>
          </h1>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
            <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest">System Ready</span>
          </div>
        </div>
      </div>

      <nav className="hidden lg:flex items-center gap-2 bg-surface/50 p-1 rounded-xl border border-white/5">
        <button 
          onClick={() => setViewMode('SANDBOX')}
          className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${viewMode === 'SANDBOX' ? 'bg-primary text-white' : 'text-slate-400 hover:text-white'}`}
        >
          SANDBOX
        </button>
        <button 
          onClick={() => setViewMode('LAB_GRID')}
          className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${viewMode === 'LAB_GRID' ? 'bg-primary text-white' : 'text-slate-400 hover:text-white'}`}
        >
          LAB GRID
        </button>
        <button 
          onClick={() => setViewMode('ANALYTICS')}
          className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${viewMode === 'ANALYTICS' ? 'bg-primary text-white' : 'text-slate-400 hover:text-white'}`}
        >
          ANALYTICS
        </button>
      </nav>

      <div className="flex items-center gap-6">
        <div className="hidden md:flex flex-col items-end">
          <span className="text-[10px] text-slate-500 uppercase">Throughput</span>
          <span className="text-sm font-mono text-primary tracking-tighter">{fps} FPS</span>
        </div>
        <div className="hidden md:flex flex-col items-end">
          <span className="text-[10px] text-slate-500 uppercase">GPU Load</span>
          <span className="text-sm font-mono text-accent tracking-tighter">{gpuLoad}%</span>
        </div>
        <button className="bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary px-4 py-2 rounded-lg flex items-center gap-2 transition-all group">
          <span className="material-icons text-sm">settings</span>
          <span className="text-sm font-semibold">Config</span>
        </button>
      </div>
    </header>
  );
};

export default Header;
