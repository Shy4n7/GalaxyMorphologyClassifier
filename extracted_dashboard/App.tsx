
import React, { useState, useEffect, useMemo, useRef } from 'react';
import CentralNode from './components/CentralNode';
import ModelCard from './components/ModelCard';
import Connections from './components/Connections';
import Background from './components/Background';
import ModelDetailsModal from './components/ModelDetailsModal';
import { DashboardState, ModelStats } from './types';

const INITIAL_MODELS: ModelStats[] = [
  {
    id: 'star-lord',
    name: 'STAR-LORD',
    description: 'Charismatic Leader',
    initial: 'S',
    color: '#ef4444',
    statusColor: 'bg-red-500',
    confidence: 94,
    accuracy: 93.8,
    sparkline: [30, 50, 40, 80, 60, 90],
    precision: 92.4,
    recall: 91.8,
    f1Score: 92.1,
    throughput: 1250,
    uptime: '99.98%'
  },
  {
    id: 'groot',
    name: 'GROOT',
    description: 'I am MobileNet',
    initial: 'G',
    color: '#fbbf24',
    statusColor: 'bg-amber-400',
    confidence: 92,
    accuracy: 92.5,
    sparkline: [20, 40, 30, 50, 45, 85],
    precision: 90.1,
    recall: 94.2,
    f1Score: 92.1,
    throughput: 4500,
    uptime: '99.95%'
  },
  {
    id: 'gamora',
    name: 'GAMORA',
    description: 'Deadly Accurate',
    initial: 'G',
    color: '#10b981',
    statusColor: 'bg-emerald-500',
    confidence: 95,
    accuracy: 94.2,
    sparkline: [70, 65, 80, 75, 85, 95],
    precision: 96.2,
    recall: 93.5,
    f1Score: 94.8,
    throughput: 850,
    uptime: '100.00%'
  },
  {
    id: 'rocket',
    name: 'ROCKET',
    description: 'Small Aggressor',
    initial: 'R',
    color: '#f97316',
    statusColor: 'bg-orange-500',
    confidence: 93,
    accuracy: 93.1,
    sparkline: [40, 60, 50, 70, 65, 88],
    precision: 91.5,
    recall: 90.2,
    f1Score: 90.8,
    throughput: 2100,
    uptime: '99.82%'
  },
  {
    id: 'drax',
    name: 'DRAX',
    description: 'Literal Power',
    initial: 'D',
    color: '#94a3b8',
    statusColor: 'bg-slate-400',
    confidence: 91,
    accuracy: 92.5,
    sparkline: [60, 70, 65, 80, 75, 92],
    precision: 89.4,
    recall: 88.7,
    f1Score: 89.0,
    throughput: 1100,
    uptime: '99.90%'
  }
];

const App: React.FC = () => {
  const [state, setState] = useState<DashboardState>({
    ensembleOutput: 95.4,
    gpuLoad: 42,
    latency: 24,
    models: INITIAL_MODELS
  });

  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [hoveredModelId, setHoveredModelId] = useState<string | null>(null);
  const [smoothScrollY, setSmoothScrollY] = useState(0);
  const scrollTargetRef = useRef(0);
  const requestRef = useRef<number>(null);

  // Constants optimized for high precision and spatial 'breath'
  const ORBIT_RADIUS = 820; 
  const ROTATION_SENSITIVITY = 0.12; 
  const TILT_ANGLE = 15 * (Math.PI / 180); 
  const LERP_FACTOR = 0.12; 

  const animate = () => {
    setSmoothScrollY((prev) => {
      const diff = scrollTargetRef.current - prev;
      if (Math.abs(diff) < 0.001) return scrollTargetRef.current;
      return prev + diff * LERP_FACTOR;
    });
    requestRef.current = requestAnimationFrame(animate);
  };

  useEffect(() => {
    requestRef.current = requestAnimationFrame(animate);
    const handleScroll = () => {
      scrollTargetRef.current = window.scrollY;
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', handleScroll);
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setState(prev => ({
        ...prev,
        ensembleOutput: Number((95.4 + (Math.random() * 0.4 - 0.2)).toFixed(1)),
        gpuLoad: Math.max(38, Math.min(48, prev.gpuLoad + (Math.random() * 4 - 2))),
        latency: Math.max(20, Math.min(30, prev.latency + (Math.random() * 2 - 1))),
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const modelPositions = useMemo(() => {
    return state.models.map((model, index) => {
      const baseAngle = (index * (360 / state.models.length)) - 90;
      const rotationAngle = baseAngle + (smoothScrollY * ROTATION_SENSITIVITY);
      const theta = (rotationAngle * Math.PI) / 180;
      
      const x = Math.cos(theta) * ORBIT_RADIUS;
      const z = Math.sin(theta) * ORBIT_RADIUS;
      const y = z * Math.sin(TILT_ANGLE); 
      const correctedZ = z * Math.cos(TILT_ANGLE); 
      
      const normalizedDepth = (correctedZ + ORBIT_RADIUS) / (2 * ORBIT_RADIUS); 
      
      const isHovered = hoveredModelId === model.id;
      const baseScale = 0.55 + (normalizedDepth * 0.45); 
      const scale = isHovered ? baseScale * 1.2 : baseScale;
      const opacity = isHovered ? 1.0 : 0.12 + (normalizedDepth * 0.88);
      const blur = isHovered ? 0 : (1 - normalizedDepth) * 15;

      return {
        ...model,
        x,
        y,
        z: correctedZ,
        scale,
        opacity,
        blur
      };
    }).sort((a, b) => a.z - b.z);
  }, [state.models, smoothScrollY, hoveredModelId]);

  const selectedModel = state.models.find(m => m.id === selectedModelId);

  return (
    <div className="relative bg-bg-dark min-h-[6000px] select-none">
      <Background scrollY={smoothScrollY} />
      
      {/* Cinematic 3D Scene - Occupies full screen without UI bars */}
      <div 
        className="fixed inset-0 flex items-center justify-center overflow-hidden pointer-events-none"
        style={{ perspective: '4000px' }}
      >
        <div 
          className="relative flex items-center justify-center" 
          style={{ 
            transformStyle: 'preserve-3d', 
            transform: `translateZ(-1000px) rotateY(${smoothScrollY * 0.0015}deg)` 
          }}
        >
          
          <Connections positions={modelPositions} hoveredModelId={hoveredModelId} />

          <div className="pointer-events-auto z-20">
            <CentralNode value={state.ensembleOutput} />
          </div>

          <div className="absolute inset-0 flex items-center justify-center" style={{ transformStyle: 'preserve-3d' }}>
            {modelPositions.map((model) => (
              <div 
                key={model.id}
                onMouseEnter={() => setHoveredModelId(model.id)}
                onMouseLeave={() => setHoveredModelId(null)}
                className="absolute pointer-events-auto will-change-transform"
                style={{
                  transform: `translate3d(${model.x}px, ${model.y}px, ${model.z}px) scale(${model.scale})`,
                  opacity: model.opacity,
                  filter: `blur(${model.blur}px)`,
                  zIndex: hoveredModelId === model.id ? 9999 : Math.round(model.z + 5000),
                  transition: 'transform 0.15s cubic-bezier(0.2, 0, 0.2, 1), opacity 0.5s ease-out, filter 0.5s ease-out',
                }}
              >
                <ModelCard 
                  stats={model} 
                  borderSide="top" 
                  onClick={() => setSelectedModelId(model.id)} 
                />
              </div>
            ))}
          </div>
        </div>
      </div>

      {selectedModel && (
        <ModelDetailsModal 
          model={selectedModel} 
          onClose={() => setSelectedModelId(null)} 
        />
      )}
    </div>
  );
};

export default App;
