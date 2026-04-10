import React from 'react';

const Header: React.FC = () => {
  return (
    <nav className="relative z-50 flex items-center justify-between px-10 py-5 glass-card border-b border-white/5">
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center text-white shadow-lg shadow-primary/20">
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3l14 9-14 9V3z" />
          </svg>
        </div>
        <span className="text-2xl font-bold tracking-tight uppercase">
          Galaxy <span className="text-primary font-black">Classifier</span>
        </span>
      </div>

      <div className="flex items-center gap-10 text-[11px] font-semibold tracking-[0.2em] uppercase text-slate-400">
        <a className="text-primary transition-colors cursor-pointer" href="#">Overview</a>
        <a className="hover:text-primary transition-colors cursor-pointer" href="http://localhost:8080/classifier" target="_blank">Classify</a>
        <a className="hover:text-primary transition-colors cursor-pointer" href="#">Models</a>
        <a className="hover:text-primary transition-colors cursor-pointer" href="#">Analytics</a>
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
  );
};

export default Header;
