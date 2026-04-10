import React from 'react';
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { ModelConfig } from '../types';

interface ModelCardProps {
  model: ModelConfig;
  onClick?: () => void;
}

const ModelCard: React.FC<ModelCardProps> = ({ model, onClick }) => {
  const initial = model.name.slice(0, 2).toUpperCase();
  const sparkline = Array.from({ length: 12 }, (_, i) =>
    Math.max(80, Math.min(100, model.accuracy + (Math.sin(i * 0.8) * 3)))
  );
  const chartData = sparkline.map((val, i) => ({ name: i.toString(), value: val }));

  return (
    <div
      onClick={onClick}
      className="relative glass-card p-7 rounded-[2.5rem] cursor-pointer group transition-all duration-700 hover:scale-[1.02] hover:-translate-y-2"
      style={{
        borderTop: `2px solid ${model.color}`,
        boxShadow: `0 12px 48px -12px ${model.color}25`,
      }}
    >
      <div
        className="absolute -inset-[2px] rounded-[2.5rem] opacity-0 group-hover:opacity-100 transition-all duration-700 pointer-events-none blur-[2px]"
        style={{ border: `1.5px solid ${model.color}`, boxShadow: `0 0 40px ${model.color}30, inset 0 0 20px ${model.color}15` }}
      />

      <div className="flex items-center gap-5 mb-6 relative z-10">
        <div
          className="w-14 h-14 rounded-[1.1rem] flex items-center justify-center text-xl font-bold border transition-all duration-700 group-hover:rounded-[2rem] group-hover:scale-110 group-hover:rotate-12"
          style={{ backgroundColor: `${model.color}15`, borderColor: `${model.color}40`, color: model.color, boxShadow: `0 0 25px ${model.color}20` }}
        >
          {initial}
        </div>
        <div className="flex-grow min-w-0">
          <h3 className="text-sm font-black tracking-widest uppercase leading-tight text-white truncate">{model.name}</h3>
          <p className="text-[9px] text-slate-500 font-mono tracking-[0.1em] font-bold uppercase group-hover:text-slate-300 transition-colors">{model.architecture}</p>
        </div>
        <div className="w-3 h-3 rounded-full flex-shrink-0 animate-pulse" style={{ backgroundColor: model.color, boxShadow: `0 0 12px ${model.color}` }} />
      </div>

      <div className="space-y-4 relative z-10">
        <div className="flex justify-between text-[10px] font-mono tracking-[0.2em] uppercase">
          <span className="text-slate-400 group-hover:text-white transition-colors">Test Accuracy (TTA)</span>
          <span className="text-slate-200 font-black">{model.accuracy}%</span>
        </div>

        <div className="h-2 w-full bg-slate-900/90 rounded-full overflow-hidden border border-white/10">
          <div
            className="h-full transition-all duration-[1500ms]"
            style={{ width: `${model.accuracy}%`, backgroundColor: model.color, boxShadow: `0 0 20px ${model.color}99` }}
          />
        </div>

        <div className="h-12 w-full relative overflow-hidden rounded-xl bg-black/20 border border-white/5 group-hover:bg-black/40 transition-all duration-700">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 4 }}>
              <YAxis domain={[75, 100]} hide />
              <Tooltip
                content={({ active, payload }) => active && payload?.length ? (
                  <div className="px-2 py-1 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg text-[10px] font-mono font-bold" style={{ borderColor: `${model.color}40`, color: model.color }}>
                    {payload[0].value}%
                  </div>
                ) : null}
                cursor={{ stroke: model.color, strokeWidth: 1, strokeDasharray: '4 4' }}
              />
              <Line type="monotone" dataKey="value" stroke={model.color} strokeWidth={2.5} dot={false} activeDot={{ r: 3, fill: model.color, stroke: 'white', strokeWidth: 2 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

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
            <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">Weight</span>
            <span className="text-[11px] font-black" style={{ color: model.color }}>{model.weight.toFixed(2)}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelCard;
