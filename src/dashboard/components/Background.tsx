
import React from 'react';

interface BackgroundProps {
  scrollY?: number;
}

const Background: React.FC<BackgroundProps> = ({ scrollY = 0 }) => {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
      {/* Pure void background with soft focal glow */}
      <div className="absolute inset-0 radial-glow opacity-40"></div>
      
      {/* Drifting ambient nebula-like particles */}
      <div 
        className="absolute top-1/4 left-1/4 w-[600px] h-[600px] bg-primary/5 rounded-full blur-[120px] animate-pulse"
        style={{ transform: `translateY(${scrollY * -0.05}px)` }}
      ></div>
      <div 
        className="absolute bottom-1/4 right-1/4 w-[800px] h-[800px] bg-blue-500/5 rounded-full blur-[150px] animate-pulse" 
        style={{ animationDelay: '2s', transform: `translateY(${scrollY * -0.08}px)` }}
      ></div>
      
      {/* Static distant field for depth */}
      <div className="absolute inset-0 opacity-10" style={{ backgroundImage: 'radial-gradient(circle, white 1px, transparent 1px)', backgroundSize: '150px 150px' }}></div>
    </div>
  );
};

export default Background;
