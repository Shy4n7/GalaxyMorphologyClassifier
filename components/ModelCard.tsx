
import React from 'react';
import { ModelConfig } from '../types';

interface ModelCardProps {
  model: ModelConfig;
}

const ModelCard: React.FC<ModelCardProps> = ({ model }) => {
  return (
    <div className="bg-surface/40 border border-primary/20 rounded-xl flex flex-col relative group overflow-hidden transition-all hover:border-primary/50">
      <div className="p-4 border-b border-primary/10 flex justify-between items-center">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wider">{model.name}</h3>
          <p className="text-[10px]" style={{ color: model.color }}>{model.architecture}</p>
        </div>
        <span className="material-icons text-lg" style={{ color: model.color }}>
          {model.id === 'resnet' ? 'layers' : model.id === 'vgg' ? 'dns' : model.id === 'inception' ? 'grid_view' : 'tune'}
        </span>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center p-6 space-y-4">
        {model.id === 'resnet' && (
          <div className="relative flex flex-col items-center gap-8">
            <div className="w-12 h-12 rounded-lg border-2 border-primary/40 bg-primary/10 flex items-center justify-center node-pulse">
              <span className="material-icons text-primary">auto_graph</span>
            </div>
            <div className="w-10 h-10 rounded-lg border border-slate-700 bg-background/80 flex items-center justify-center">
              <span className="text-[10px] text-slate-500 font-mono">CONV</span>
            </div>
            <div className="w-12 h-12 rounded-lg border-2 border-primary bg-primary/20 flex items-center justify-center node-pulse">
              <span className="material-icons text-primary">add_circle_outline</span>
            </div>
          </div>
        )}

        {model.id === 'vgg' && (
          <div className="flex flex-col gap-2 items-center">
            {[1, 2, 3, 4, 5].map(b => (
              <div key={b} className="w-20 h-6 rounded bg-accent/20 border border-accent/40 flex items-center justify-center">
                <span className="text-[8px] text-accent font-bold">BLOCK {b}</span>
              </div>
            ))}
            <div className="w-24 h-8 rounded border-2 border-accent bg-accent/10 flex items-center justify-center node-pulse mt-2">
              <span className="text-[10px] text-white font-bold uppercase">Flatten</span>
            </div>
          </div>
        )}

        {model.id === 'inception' && (
          <div className="grid grid-cols-3 gap-2 w-full">
            <div className="col-span-3 h-4 bg-emerald-500/10 border border-emerald-500/30 rounded"></div>
            <div className="h-16 bg-emerald-500/20 border-2 border-emerald-500/40 rounded flex flex-col items-center justify-center">
              <span className="text-[8px] text-emerald-500">1x1</span>
            </div>
            <div className="h-16 bg-emerald-500/20 border-2 border-emerald-500/40 rounded flex flex-col items-center justify-center node-pulse">
              <span className="text-[8px] text-emerald-500">3x3</span>
            </div>
            <div className="h-16 bg-emerald-500/20 border-2 border-emerald-500/40 rounded flex flex-col items-center justify-center">
              <span className="text-[8px] text-emerald-500">5x5</span>
            </div>
            <div className="col-span-3 h-8 bg-emerald-500/40 border-2 border-emerald-500 rounded flex items-center justify-center">
              <span className="text-[10px] font-bold text-white uppercase tracking-tighter">Concatenate</span>
            </div>
          </div>
        )}

        {model.id === 'custom' && (
           <div className="w-full space-y-3">
              <div className="space-y-1">
                <div className="flex justify-between text-[8px] text-slate-500">
                  <span>Dropout</span>
                  <span className="text-amber-500">0.25</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 w-[25%]"></div>
                </div>
              </div>
              <div className="space-y-1">
                <div className="flex justify-between text-[8px] text-slate-500">
                  <span>LR</span>
                  <span className="text-amber-500">1e-4</span>
                </div>
                <div className="h-1 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-amber-500 w-[40%]"></div>
                </div>
              </div>
              <div className="grid grid-cols-4 gap-1">
                {[1,2,3,4,5,6,7,8].map(i => <div key={i} className="h-4 bg-amber-500/10 border border-amber-500/30 rounded"></div>)}
              </div>
           </div>
        )}
      </div>

      <div className="p-3 bg-background/60 text-[10px] font-mono text-slate-500 flex justify-between border-t border-primary/10">
        <span>PARAMS: {model.params}</span>
        <span>FLOPs: {model.flops}</span>
      </div>
    </div>
  );
};

export default ModelCard;
