
import React, { useEffect, useState, useCallback } from 'react';
import { ModelStats } from '../types';

interface ModelDetailsModalProps {
  model: ModelStats;
  onClose: () => void;
}

const ModelDetailsModal: React.FC<ModelDetailsModalProps> = ({ model, onClose }) => {
  const [isVisible, setIsVisible] = useState(false);

  const handleClose = useCallback(() => {
    setIsVisible(false);
    // Wait for the exit animation to complete before calling onClose
    setTimeout(onClose, 500);
  }, [onClose]);

  useEffect(() => {
    // Small delay to trigger entry animation
    const timer = setTimeout(() => setIsVisible(true), 20);
    
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        handleClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    
    return () => {
      clearTimeout(timer);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleClose]);

  // Mock historical data for a larger chart
  const historicalAccuracy = [
    88.2, 89.5, 88.9, 91.2, 90.5, 92.8, 93.1, 92.5, 94.0, 93.8
  ];

  return (
    <div 
      className={`fixed inset-0 z-[100] flex items-center justify-center p-6 transition-all duration-700 cubic-bezier(0.16, 1, 0.3, 1) ${isVisible ? 'opacity-100' : 'opacity-0'}`}
    >
      <style>{`
        .cubic-bezier-premium {
          transition-timing-function: cubic-bezier(0.16, 1, 0.3, 1);
        }
      `}</style>

      {/* Backdrop with synchronized blur transition */}
      <div 
        className={`absolute inset-0 bg-bg-dark/40 transition-all duration-700 cubic-bezier-premium ${isVisible ? 'backdrop-blur-2xl bg-bg-dark/80' : 'backdrop-blur-0 bg-bg-dark/0'}`} 
        onClick={handleClose}
      />

      {/* Modal Content */}
      <div 
        className={`relative w-full max-w-4xl glass-card rounded-[2.5rem] overflow-hidden border-2 transition-all duration-700 cubic-bezier-premium transform will-change-transform ${
          isVisible 
            ? 'translate-y-0 scale-100 rotate-0' 
            : 'translate-y-20 scale-90 -rotate-1'
        }`} 
        style={{ 
          borderColor: `${model.color}40`,
          boxShadow: isVisible 
            ? `0 40px 120px -20px ${model.color}30, 0 0 40px -10px ${model.color}20` 
            : '0 0 0 0 transparent'
        }}
      >
        
        {/* Header */}
        <div className="px-10 py-8 border-b border-white/5 flex items-center justify-between relative overflow-hidden">
          {/* Subtle background glow for the header */}
          <div className="absolute top-0 right-0 w-64 h-64 opacity-10 blur-3xl rounded-full pointer-events-none" style={{ backgroundColor: model.color }}></div>
          
          <div className="flex items-center gap-8 relative z-10">
            <div 
              className={`w-20 h-20 rounded-2xl flex items-center justify-center text-3xl font-bold border-2 transition-all duration-1000 delay-100 ${isVisible ? 'scale-100 rotate-0 opacity-100' : 'scale-50 rotate-12 opacity-0'}`} 
              style={{ 
                backgroundColor: `${model.color}15`, 
                borderColor: `${model.color}60`,
                color: model.color 
              }}
            >
              {model.initial}
            </div>
            <div>
              <h2 className="text-4xl font-black tracking-widest uppercase text-white">
                {model.name} <span className="text-slate-500 font-light text-xl">v2.4.0</span>
              </h2>
              <div className="flex items-center gap-5 mt-2">
                <span className="text-xs font-mono text-slate-400 tracking-[0.2em] uppercase font-bold">{model.description}</span>
                <span className="text-[10px] px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-black tracking-widest uppercase">Cluster Alpha-9</span>
              </div>
            </div>
          </div>
          
          <button 
            onClick={handleClose}
            className="group relative p-3 text-slate-400 hover:text-white transition-all bg-white/5 hover:bg-white/10 rounded-full active:scale-90"
            aria-label="Close modal"
          >
            <svg className="w-8 h-8 transition-transform group-hover:rotate-90 duration-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content Area */}
        <div className="p-10 grid grid-cols-12 gap-10">
          
          {/* Main Chart Section */}
          <div className="col-span-8 space-y-8">
            <div className={`glass-card bg-white/[0.03] rounded-[2rem] p-8 border border-white/5 transition-all duration-1000 delay-200 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
              <div className="flex items-center justify-between mb-10">
                <h3 className="text-xs font-black tracking-[0.3em] uppercase text-slate-400">Accuracy Timeline (Last 24h)</h3>
                <div className="flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full animate-pulse" style={{ backgroundColor: model.color }}></span>
                  <span className="text-[10px] font-mono text-slate-500 uppercase tracking-widest font-bold">Real-time Telemetry</span>
                </div>
              </div>
              
              <div className="relative h-56 w-full">
                <svg className="w-full h-full overflow-visible" preserveAspectRatio="none">
                  <defs>
                    <linearGradient id={`modalGrad-${model.id}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={model.color} stopOpacity="0.4" />
                      <stop offset="100%" stopColor={model.color} stopOpacity="0" />
                    </linearGradient>
                  </defs>
                  
                  {/* Grid Lines */}
                  {[0, 1, 2, 3].map(i => (
                    <line key={i} x1="0" y1={`${i * 33.3}%`} x2="100%" y2={`${i * 33.3}%`} stroke="white" strokeOpacity="0.05" strokeWidth="1" />
                  ))}

                  {/* Area */}
                  <path 
                    d={`M 0,224 ${historicalAccuracy.map((v, i) => `L ${i * (100 / 9)}%,${224 - ((v - 85) * 18)}`).join(' ')} L 100,224 Z`} 
                    fill={`url(#modalGrad-${model.id})`}
                    className="transition-all duration-1000 ease-out"
                  />

                  {/* Line */}
                  <path 
                    d={`M 0,${224 - ((historicalAccuracy[0] - 85) * 18)} ${historicalAccuracy.map((v, i) => `L ${i * (100 / 9)}%,${224 - ((v - 85) * 18)}`).join(' ')}`} 
                    fill="none" 
                    stroke={model.color} 
                    strokeWidth="4" 
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    className="transition-all duration-1000 ease-out"
                  />
                  
                  {/* Points */}
                  {historicalAccuracy.map((v, i) => (
                    <circle 
                      key={i} 
                      cx={`${i * (100 / 9)}%`} 
                      cy={`${224 - ((v - 85) * 18)}`} 
                      r="5" 
                      fill={model.color} 
                      stroke="#020617" 
                      strokeWidth="2.5" 
                      className="transition-all duration-500 hover:scale-150 cursor-crosshair"
                    />
                  ))}
                </svg>
              </div>
              
              <div className="flex justify-between mt-6 text-[10px] font-mono text-slate-500 uppercase tracking-[0.3em] font-bold">
                <span>-24 Hours</span>
                <span className="opacity-30">T-Minus Midpoint</span>
                <span className="text-white opacity-80">Synchronized Now</span>
              </div>
            </div>

            {/* Logs Preview */}
            <div className={`glass-card bg-black/50 rounded-[1.5rem] p-6 border border-white/5 font-mono text-[11px] h-40 overflow-hidden transition-all duration-1000 delay-300 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
              <div className="text-primary mb-4 opacity-50 uppercase tracking-[0.2em] font-bold">System Logs Console</div>
              <div className="space-y-2 text-slate-400">
                <div className="flex gap-5">
                  <span className="text-slate-600 font-light">[21:44:02]</span>
                  <span className="tracking-tight">INF: Weight update initialized from Ensemble Orchestrator.</span>
                </div>
                <div className="flex gap-5">
                  <span className="text-slate-600 font-light">[21:44:05]</span>
                  <span className="text-emerald-500 font-bold">RES: Model {model.name} responded with {model.confidence}% confidence.</span>
                </div>
                <div className="flex gap-5 text-slate-500">
                  <span className="text-slate-600 font-light">[21:44:09]</span>
                  <span className="italic">DEBUG: Jitter buffer cleared in 12ms (Target: 20ms).</span>
                </div>
                <div className="flex gap-5">
                  <span className="text-slate-600 font-light">[21:44:12]</span>
                  <span className="tracking-tight">INF: New batch processing (size: 256) started on node 0x82.</span>
                </div>
              </div>
            </div>
          </div>

          {/* Metrics Sidebar */}
          <div className="col-span-4 space-y-5">
            {[
              { label: "Precision", value: `${model.precision}%`, color: model.color, delay: 400 },
              { label: "Recall", value: `${model.recall}%`, color: model.color, delay: 450 },
              { label: "F1-Score", value: `${model.f1Score.toFixed(1)}`, color: model.color, delay: 500 },
              { label: "Throughput", value: `${model.throughput} req/s`, color: model.color, delay: 550 },
              { label: "Uptime", value: model.uptime, color: "#10b981", delay: 600 }
            ].map((metric, idx) => (
              <div 
                key={metric.label}
                className={`glass-card p-5 rounded-2xl border border-white/5 flex flex-col items-center justify-center group hover:bg-white/[0.05] hover:border-white/20 transition-all duration-700 ${isVisible ? 'translate-x-0 opacity-100' : 'translate-x-10 opacity-0'}`}
                style={{ transitionDelay: `${metric.delay}ms` }}
              >
                <div className="text-[10px] font-mono text-slate-500 uppercase tracking-[0.3em] font-bold mb-2 group-hover:text-slate-300 transition-colors">{metric.label}</div>
                <div className="text-3xl font-black tracking-tight transition-all group-hover:scale-110" style={{ color: metric.color }}>{metric.value}</div>
              </div>
            ))}
            
            <button className={`w-full mt-6 py-5 rounded-2xl border border-white/10 bg-white/5 hover:bg-white/10 hover:border-white/30 transition-all duration-700 font-black text-[11px] tracking-[0.4em] uppercase text-white/60 hover:text-white ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`} style={{ transitionDelay: '700ms' }}>
              Full Analytics Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetailsModal;
