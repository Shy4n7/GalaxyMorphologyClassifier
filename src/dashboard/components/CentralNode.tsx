
import React from 'react';

interface CentralNodeProps {
  value: number;
}

const CentralNode: React.FC<CentralNodeProps> = ({ value }) => {
  return (
    <div className="relative w-80 h-80 flex flex-col items-center justify-center rounded-full glass-card border-2 border-primary/30 shadow-[0_0_80px_rgba(59,130,246,0.2)] node-glow pulse">
      {/* Volumetric decorative rings */}
      <div className="absolute inset-[-60px] rounded-full border border-primary/5 pointer-events-none animate-[spin_20s_linear_infinite]"></div>
      <div className="absolute inset-[-30px] rounded-full border border-dashed border-primary/10 pointer-events-none animate-[spin_12s_linear_infinite_reverse]"></div>
      <div className="absolute inset-0 rounded-full bg-primary/5 blur-3xl -z-10"></div>

      <div className="text-[10px] font-mono text-primary mb-3 tracking-[0.4em] font-bold uppercase opacity-60">
        Global Accuracy
      </div>
      
      <div className="flex items-baseline leading-none group cursor-default">
        <span className="text-8xl font-black tracking-tighter text-white drop-shadow-[0_0_15px_rgba(255,255,255,0.2)]">
          {value.toFixed(1).split('.')[0]}
        </span>
        <span className="text-6xl font-black tracking-tighter text-white/90">
          .{value.toFixed(1).split('.')[1]}
        </span>
        <span className="text-2xl font-bold text-primary ml-1 opacity-70">%</span>
      </div>

      <div className="mt-8 flex flex-col items-center">
        <div className="h-1 w-24 bg-primary/20 rounded-full mb-4 overflow-hidden">
          <div className="h-full bg-primary animate-[shimmer_2s_infinite]" style={{ width: '60%' }}></div>
        </div>
        <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">Ensemble Synced</span>
      </div>

      <style>{`
        @keyframes shimmer {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(200%); }
        }
      `}</style>
    </div>
  );
};

export default CentralNode;
