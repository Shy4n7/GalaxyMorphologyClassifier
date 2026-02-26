
import React from 'react';

interface FooterProps {
  gpuLoad: number;
  latency: number;
}

const Footer: React.FC<FooterProps> = ({ gpuLoad, latency }) => {
  return (
    <footer className="relative z-50 px-10 py-4 glass-card border-t border-white/5 flex items-center justify-between">
      <div className="flex items-center gap-12">
        <div className="flex items-center gap-4 text-[11px] font-mono tracking-widest text-slate-400 uppercase">
          <span className="opacity-60">GPU Load</span>
          <div className="flex gap-1.5">
            {[...Array(5)].map((_, i) => (
              <div 
                key={i} 
                className={`w-1 h-3 rounded-full transition-colors duration-500 ${i < Math.floor(gpuLoad/20) + 1 ? 'bg-primary shadow-[0_0_8px_rgba(59,130,246,0.6)]' : 'bg-slate-800'}`} 
              />
            ))}
          </div>
          <span className="text-primary font-bold">{Math.round(gpuLoad)}%</span>
        </div>

        <div className="flex items-center gap-4 text-[11px] font-mono tracking-widest text-slate-400 uppercase">
          <span className="opacity-60">Latency</span>
          <span className="text-emerald-500 font-bold">{Math.round(latency)}ms</span>
        </div>
      </div>

      <div className="text-[11px] font-mono text-slate-500 italic tracking-wider">
        Last re-weighting operation: 2s ago
      </div>
    </footer>
  );
};

export default Footer;
