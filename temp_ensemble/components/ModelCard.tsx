
import React from 'react';
import { LineChart, Line, ResponsiveContainer, Tooltip, YAxis } from 'recharts';
import { ModelStats } from '../types';

interface ModelCardProps {
  stats: ModelStats;
  borderSide?: 'top' | 'none';
  onClick?: () => void;
}

const CustomTooltip = ({ active, payload, color }: any) => {
  if (active && payload && payload.length) {
    return (
      <div 
        className="px-2 py-1 bg-black/80 backdrop-blur-md border border-white/10 rounded-lg shadow-xl"
        style={{ borderColor: `${color}40` }}
      >
        <p className="text-[10px] font-mono font-bold tracking-widest text-white uppercase">
          Val: <span style={{ color }}>{payload[0].value}%</span>
        </p>
      </div>
    );
  }
  return null;
};

const ModelCard: React.FC<ModelCardProps> = ({ stats, borderSide = 'top', onClick }) => {
  // Format sparkline data for Recharts
  const chartData = stats.sparkline.map((val, index) => ({
    name: index.toString(),
    value: val,
  }));

  return (
    <div 
      onClick={onClick}
      className={`
        relative w-80 glass-card p-7 rounded-[2.5rem] cursor-pointer group 
        transition-all duration-1000 cubic-bezier(0.16, 1, 0.3, 1)
        hover:bg-white/[0.1] active:scale-95
        ${borderSide === 'top' ? 'border-t-2' : ''}
      `} 
      style={{ 
        borderTopColor: stats.color,
        boxShadow: `0 12px 48px -12px ${stats.color}25`,
      }}
    >
      {/* Perfect Round Hover Border */}
      <div 
        className="absolute -inset-[2px] rounded-[2.5rem] opacity-0 group-hover:opacity-100 transition-all duration-1000 pointer-events-none blur-[2px]"
        style={{ 
          border: `1.5px solid ${stats.color}`,
          boxShadow: `0 0 40px ${stats.color}30, inset 0 0 20px ${stats.color}15`
        }}
      />

      <div className="flex items-center gap-5 mb-6 relative z-10">
        <div 
          className="w-14 h-14 rounded-[1.1rem] flex items-center justify-center text-xl font-bold border transition-all duration-1000 group-hover:rounded-[2rem] group-hover:scale-110 group-hover:rotate-[12deg]" 
          style={{ 
            backgroundColor: `${stats.color}15`, 
            borderColor: `${stats.color}40`,
            color: stats.color,
            boxShadow: `0 0 25px ${stats.color}20`
          }}
        >
          {stats.initial}
        </div>
        <div className="flex-grow">
          <h3 className="text-xl font-black tracking-widest uppercase leading-tight text-white transition-all duration-700 group-hover:tracking-[0.15em]">
            {stats.name}
          </h3>
          <p className="text-[9px] text-slate-500 font-mono tracking-[0.1em] font-bold uppercase group-hover:text-slate-300 transition-colors">
            {stats.description}
          </p>
        </div>
        <div 
          className={`w-3 h-3 rounded-full ${stats.statusColor} shadow-[0_0_20px_${stats.color}] group-hover:scale-150 transition-all duration-700`} 
        />
      </div>

      <div className="space-y-4 relative z-10">
        <div className="flex justify-between text-[10px] font-mono tracking-[0.2em] uppercase">
          <span className="text-slate-400 group-hover:text-white transition-colors">Confidence</span>
          <span className="text-slate-200 font-black group-hover:text-white">{stats.accuracy}% ACC</span>
        </div>
        
        <div className="h-2 w-full bg-slate-900/90 rounded-full overflow-hidden border border-white/10">
          <div 
            className="h-full transition-all duration-[1500ms] cubic-bezier(0.16, 1, 0.3, 1)" 
            style={{ 
              width: `${stats.confidence}%`, 
              backgroundColor: stats.color,
              boxShadow: `0 0 20px ${stats.color}99`
            }} 
          />
        </div>

        {/* Enhanced Sparkline */}
        <div className="h-12 w-full relative overflow-hidden rounded-xl bg-black/20 border border-white/5 transition-all duration-700 group-hover:bg-black/40 group-hover:border-white/10">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 4 }}>
              <Tooltip 
                content={<CustomTooltip color={stats.color} />} 
                cursor={{ stroke: stats.color, strokeWidth: 1, strokeDasharray: '4 4' }}
                isAnimationActive={false}
              />
              <YAxis domain={[0, 100]} hide />
              <Line 
                type="monotone" 
                dataKey="value" 
                stroke={stats.color} 
                strokeWidth={2.5} 
                dot={false}
                activeDot={{ r: 3, fill: stats.color, stroke: 'white', strokeWidth: 2 }}
                animationDuration={2000}
                isAnimationActive={true}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* New Metrics Snapshot Row */}
        <div className="flex items-center justify-between pt-2 border-t border-white/5 mt-2">
           <div className="flex flex-col items-start">
              <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">Prec.</span>
              <span className="text-[11px] font-black text-white">{stats.precision}%</span>
           </div>
           <div className="flex flex-col items-center">
              <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">Recall</span>
              <span className="text-[11px] font-black text-white">{stats.recall}%</span>
           </div>
           <div className="flex flex-col items-end">
              <span className="text-[8px] font-mono text-slate-500 uppercase font-bold">F1</span>
              <span className="text-[11px] font-black" style={{ color: stats.color }}>{stats.f1Score.toFixed(1)}</span>
           </div>
        </div>
      </div>

      <style>{`
        .glass-card:hover {
          box-shadow: 0 40px 100px -20px ${stats.color}50, 0 0 60px -10px ${stats.color}25 !important;
          transform: translateY(-10px) scale(1.02) !important;
        }
      `}</style>
    </div>
  );
};

export default ModelCard;
