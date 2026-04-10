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
    <nav className="relative z-50 flex items-center justify-between px-10 py-5 glass-card border-b border-white/5">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center text-white shadow-lg shadow-primary/20">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <span className="text-2xl font-bold tracking-tight uppercase">
          Ensemble <span className="text-primary font-black">OS</span>
        </span>
      </div>

      <div className="flex items-center gap-10 text-[11px] font-semibold tracking-[0.2em] uppercase text-slate-400">
        {(['SANDBOX', 'LAB_GRID', 'ANALYTICS'] as ViewMode[]).map(v => (
          <button
            key={v}
            onClick={() => setViewMode(v)}
            className={`transition-colors cursor-pointer ${viewMode === v ? 'text-primary' : 'hover:text-primary'}`}
          >
            {v === 'SANDBOX' ? 'Classify' : v === 'LAB_GRID' ? 'Lab Grid' : 'Analytics'}
          </button>
        ))}
      </div>

      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
          <span className="text-slate-500">FPS</span>
          <span className="text-white font-bold">{fps}</span>
        </div>
        <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
          <span className="text-slate-500">GPU</span>
          <span className="text-white font-bold">{gpuLoad}%</span>
        </div>
        <div className="flex items-center gap-2.5 px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20 text-green-400 text-[10px] font-mono font-bold">
          <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></span>
          SYSTEM ACTIVE
        </div>
      </div>
    </nav>
  );
};

export default Header;
