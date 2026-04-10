import React from 'react';
import { ModelConfig } from '../types';

interface ModelCardProps {
  model: ModelConfig;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  const initial = model.name.slice(0, 2).toUpperCase();

  return (
    <div
      className="relative glass-card p-7 rounded-[2.5rem] group transition-all duration-700 hover:scale-[1.02] hover:-translate-y-2 cursor-default"
      style={{
        borderTop: `2px solid ${model.color}`,
        boxShadow: `0 12px 48px -12px ${model.color}25`,
      }}
    >
      {/* Hover border glow */}
      <div
        className="absolute -inset-[2px] rounded-[2.5rem] opacity-0 group-hover:opacity-100 transition-all duration-700 pointer-events-none blur-[2px]"
        style={{ border: `1.5px solid ${model.color}`, boxShadow: `0 0 40px ${model.color}30, inset 0 0 20px ${model.color}15` }}
      />

      {/* Header */}
      <div className="flex items-center gap-4 mb-6 relative z-10">
        <div
          className="w-14 h-14 rounded-[1.1rem] flex items-center justify-center text-xl font-bold border transition-all duration-700 group-hover:rounded-[2rem] group-hover:scale-110 group-hover:rotate-12"
          style={{ backgroundColor: `${model.color}15`, borderColor: `${model.color}40`, color: model.color, boxShadow: `0 0 25px ${model.color}20` }}
        >
          {initial}
        </div>
        <div className="flex-grow min-w-0">
          <h3 className="text-sm font-black tracking-widest uppercase leading-tight text-white truncate">
            {model.name}
          </h3>
          <p className="text-[9px] text-slate-500 font-mono tracking-[0.1em] font-bold uppercase group-hover:text-slate-300 transition-colors">
            {model.architecture}
          </p>
        </div>
        <div className="w-2.5 h-2.5 rounded-full animate-pulse flex-shrink-0" style={{ backgroundColor: model.color, boxShadow: `0 0 12px ${model.color}` }} />
      </div>

      {/* Accuracy bar */}
      <div className="space-y-3 relative z-10">
        <div className="flex justify-between text-[10px] font-mono tracking-[0.15em] uppercase">
          <span className="text-slate-400 group-hover:text-white transition-colors">Test Accuracy (TTA)</span>
          <span className="text-slate-200 font-black">{model.accuracy}%</span>
        </div>
        <div className="h-2 w-full bg-slate-900/90 rounded-full overflow-hidden border border-white/10">
          <div
            className="h-full transition-all duration-1500"
            style={{ width: `${model.accuracy}%`, backgroundColor: model.color, boxShadow: `0 0 20px ${model.color}99` }}
          />
        </div>

        {/* Orbit ring visual — unique to this 3-model design */}
        <div className="flex items-center justify-center py-4 relative">
          <div
            className="w-20 h-20 rounded-full border-2 flex items-center justify-center relative group-hover:scale-110 transition-all duration-700"
            style={{ borderColor: `${model.color}40` }}
          >
            <div
              className="absolute inset-0 rounded-full border border-dashed animate-spin"
              style={{ borderColor: `${model.color}20`, animationDuration: '8s' }}
            />
            <span className="text-lg font-black" style={{ color: model.color }}>
              {model.weight.toFixed(2)}
            </span>
          </div>
          <div className="absolute left-0 right-0 text-center mt-16 text-[8px] text-slate-600 uppercase font-bold tracking-widest pt-1">
            weight
          </div>
        </div>

        {/* Stats row */}
        <div className="flex items-center justify-between pt-2 border-t border-white/5">
          <div className="flex flex-col items-start">
            <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">Params</span>
            <span className="text-[11px] font-black text-white">{model.params}</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">FLOPs</span>
            <span className="text-[11px] font-black text-white">{model.flops}</span>
          </div>
          <div className="flex flex-col items-end">
            <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">Rank</span>
            <span className="text-[11px] font-black" style={{ color: model.color }}>
              #{['convnext', 'resnext50', 'densenet161'].indexOf(model.id) + 1}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelCard;
