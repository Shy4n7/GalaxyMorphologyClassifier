
import React from 'react';

interface Position {
  id: string;
  x: number;
  y: number;
  z: number;
  opacity: number;
  color: string;
}

interface ConnectionsProps {
  positions: Position[];
  hoveredModelId: string | null;
}

const Connections: React.FC<ConnectionsProps> = ({ positions, hoveredModelId }) => {
  return (
    <svg className="absolute inset-0 w-[250vw] h-[250vh] pointer-events-none z-0 overflow-visible" style={{ left: '-75vw', top: '-75vh' }}>
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feComposite in="SourceGraphic" in2="blur" operator="over" />
        </filter>
      </defs>
      
      <g transform="translate(125vw, 125vh)">
        {positions.map((pos) => {
          const isHovered = hoveredModelId === pos.id;
          // Smoothly scale connection intensity based on depth if not hovered
          const strokeOpacity = isHovered ? 1.0 : pos.opacity * 0.2;
          const strokeWidth = isHovered ? 3 : 1;
          
          return (
            <g key={pos.id} className="transition-all duration-1000 cubic-bezier(0.16, 1, 0.3, 1)">
              {/* Highlight Background Line (Atmospheric Tether) */}
              {isHovered && (
                <line 
                  x1="0" 
                  y1="0" 
                  x2={pos.x} 
                  y2={pos.y} 
                  stroke={pos.color} 
                  strokeWidth={strokeWidth * 6} 
                  strokeOpacity="0.15"
                  filter="url(#glow)"
                  className="transition-all duration-700"
                />
              )}
              
              {/* Main Data Tether */}
              <line 
                x1="0" 
                y1="0" 
                x2={pos.x} 
                y2={pos.y} 
                stroke={isHovered ? pos.color : "#3b82f6"} 
                strokeWidth={strokeWidth} 
                strokeDasharray={isHovered ? "0" : "6 18"}
                strokeOpacity={strokeOpacity}
                className={!isHovered ? "animate-[dash_15s_linear_infinite]" : "transition-all duration-500"}
              />
              
              {/* Inner High-Energy Stream */}
              {isHovered && (
                <line 
                  x1="0" 
                  y1="0" 
                  x2={pos.x} 
                  y2={pos.y} 
                  stroke="white" 
                  strokeWidth="1" 
                  strokeOpacity="0.4"
                  className="animate-pulse"
                />
              )}
            </g>
          );
        })}
      </g>
      <style>{`
        @keyframes dash {
          to {
            stroke-dashoffset: -240;
          }
        }
      `}</style>
    </svg>
  );
};

export default Connections;
